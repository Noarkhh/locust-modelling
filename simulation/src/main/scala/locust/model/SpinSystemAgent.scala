package pl.edu.agh.locust.model

import breeze.linalg.{DenseVector, norm, normalize, sum, min, max}
import breeze.numerics.{acos, cos, sin, atan2}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import pl.edu.agh.locust.utils.Xorshift32
import scala.math.Pi
import breeze.numerics.sqrt
import breeze.numerics.exp
import breeze.numerics.pow
import breeze.numerics.abs
import breeze.linalg.DenseMatrix

final case class SpinSystemAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    id: Long,
    speed: Double,
    var neuronSpinStates: DenseVector[Double]
) extends ParticleAgent

object SpinSystemAgent {
  private var allocentricNeuronAngles: Seq[Double] = Seq.empty
  private val rngState: ThreadLocal[Int] =
    ThreadLocal.withInitial(() => scala.util.Random.nextInt())

  private var synapticConnectivityMatrix = DenseMatrix.zeros[Double](1, 1)

  def init()(implicit config: ParticleAgentConfig) = {
    val neuronsAngleStep = (2 * Pi) / config.neuronsAmount
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep)

    synapticConnectivityMatrix = DenseMatrix.tabulate(config.neuronsAmount, config.neuronsAmount)({
      case (i, j) => {
        if (i == j) 0.0
        else {
          val angleBetweenNeurons =
            Pi - abs(Pi - abs(allocentricNeuronAngles(i) - allocentricNeuronAngles(j)))

          cos(Pi * pow(angleBetweenNeurons / Pi, config.synapticConnectivityCoefficient))
        }
      }
    })

  }

  def apply(
      position: DenseVector[Double],
      direction: DenseVector[Double],
      id: Long
  )(implicit config: ParticleAgentConfig): SpinSystemAgent = {
    val neuronSpinStates =
      DenseVector.tabulate(config.neuronsAmount)(i =>
        if (config.random.nextBoolean()) 1.0 else -1.0
      )
    SpinSystemAgent(position, direction, id, config.averageSpeed, neuronSpinStates)
  }

  implicit case object Behaviour extends AgentBehaviour[SpinSystemAgent] {

    override def update(agent: SpinSystemAgent, others: Iterable[SpinSystemAgent])(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      val targets: Iterable[(DenseVector[Double], Double)] = others.map(other => {
        val vectorToOther = other.position - agent.position
        val distanceToOther = vectorToOther.norm()
        val directionToOther = vectorToOther.normalize()

        (directionToOther, distanceToOther)
      })

      val referenceVector =
        if (config.allocentricReferenceFrame) DenseVector[Double](1.0, 0.0) else agent.direction

      val egocentricNeuronDirections = allocentricNeuronAngles.map(rotateVector(referenceVector, _))

      val neuronsExternalStimuli =
        egocentricNeuronDirections
          .map(egocentricNeuronDirection => {
            targets
              .map({ case (directionToOther, distanceToOther) =>
                val neuronTargetAngle = acos(egocentricNeuronDirection.dot(directionToOther))

                (
                  config.externalStimulusStrength /
                    sqrt(2 * Pi * config.receptiveFieldVariance)
                ) * exp(
                  -pow(-neuronTargetAngle, 2) /
                    (2 * config.receptiveFieldVariance)
                )
              })
              .sum
          })
          .toArray

      runNeuralDynamics(agent.neuronSpinStates, neuronsExternalStimuli, agent.id)

      val neuralForce = egocentricNeuronDirections
        .zip(agent.neuronSpinStates.toArray)
        .filter({ case (direction, spin) => spin > 0 })
        .map({ case (direction, spin) => direction })
        .reduce(_ + _)

      val velocity = (config.averageSpeed / config.neuronsAmount) * neuralForce

      agent.copy(direction = velocity.normalize(), speed = velocity.norm())

    }

    override def move(agent: SpinSystemAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      val newPosition = agent.position + agent.direction * agent.speed * deltaTime

      agent.copy(position = newPosition)
    }

    override def translate(
        agent: SpinSystemAgent,
        newPosition: DenseVector[Double]
    ): SpinSystemAgent = {
      agent.copy(position = newPosition)
    }

    override def getSpeed(agent: SpinSystemAgent)(implicit config: ParticleAgentConfig): Double =
      config.averageSpeed

    private def rotateVector(vector: DenseVector[Double], angle: Double): DenseVector[Double] =
      DenseVector[Double](
        cos(angle) * vector(0) - sin(angle) * vector(1),
        sin(angle) * vector(0) + cos(angle) * vector(1)
      )

    private def runNeuralDynamics(
        neuronSpinStates: DenseVector[Double],
        externalStimuli: Seq[Double],
        agentId: Long
    )(implicit config: ParticleAgentConfig) = {
      for (iteration <- 0 until config.neuralDynamicIterationsPerTimestep) {
        val neuronToFlip = Xorshift32.nextInt(rngState, config.neuronsAmount)

        val flipProbability = exp(
          -config.inverseTemperatureCoefficient *
            calculateDeltaHamiltonian(neuronToFlip, neuronSpinStates, externalStimuli)
        )
        if (Xorshift32.nextFloat(rngState) < flipProbability) neuronSpinStates(neuronToFlip) *= -1
      }
    }

    private def calculateDeltaHamiltonian(
        selectedNeuron: Int,
        neuronSpinStates: DenseVector[Double],
        externalStimuli: Seq[Double]
    )(implicit config: ParticleAgentConfig): Double = {
      (2 * neuronSpinStates(selectedNeuron)) *
        (
          sum(synapticConnectivityMatrix(::, selectedNeuron) *:* neuronSpinStates)
            / config.neuronsAmount +
            externalStimuli(selectedNeuron) -
            config.neuralInhibitionCoefficient
        )
    }
  }
}
