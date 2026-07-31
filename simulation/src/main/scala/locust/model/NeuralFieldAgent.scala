package pl.edu.agh.locust.model

import breeze.linalg.{DenseVector, DenseMatrix, Axis, *, norm, normalize, sum, min, max}
import breeze.numerics.{acos, cos, sin, atan2, sqrt, exp, pow, abs, tanh}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import pl.edu.agh.locust.utils.Xorshift32
import scala.math.Pi
import breeze.linalg.operators.OpMulScalar

final case class NeuralFieldAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    id: Long,
    speed: Double,
    membranePotentials: DenseVector[Double]
) extends ParticleAgent

object NeuralFieldAgent {
  private var allocentricNeuronAngles: Array[Double] = Array.empty
  private val rngState: ThreadLocal[Int] =
    ThreadLocal.withInitial(() => scala.util.Random.nextInt())

  private var synapticConnectivityMatrix = DenseMatrix.zeros[Double](1, 1)

  def init()(implicit config: ParticleAgentConfig) = {
    val neuronsAngleStep = (2 * Pi) / config.neuronsAmount
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep).toArray

    synapticConnectivityMatrix =
      DenseMatrix.tabulate(config.neuronsAmount, config.neuronsAmount)({ case (i, j) =>
        val angleBetweenNeurons =
          Pi - abs(Pi - abs(allocentricNeuronAngles(i) - allocentricNeuronAngles(j)))
        cos(Pi * pow(angleBetweenNeurons / Pi, config.synapticConnectivityCoefficient))
      })
  }

  def apply(
      position: DenseVector[Double],
      direction: DenseVector[Double],
      id: Long
  )(implicit config: ParticleAgentConfig): NeuralFieldAgent = {
    // val membranePotentials =
    //   DenseVector.tabulate(config.neuronsAmount)(i => 0.1 * config.random.nextGaussian())
    val membranePotentials = new DenseVector(
      allocentricNeuronAngles.map(neuronAngle => {
        val headingAngle = atan2(direction(1), direction(0))
        val angleBias = abs(Pi - abs(headingAngle - neuronAngle)) / Pi
        config.random.nextGaussian() * 0.75 + angleBias * 0.25
      })
    )
    NeuralFieldAgent(position, direction, id, config.averageSpeed, membranePotentials)
  }

  implicit case object Behaviour extends AgentBehaviour[NeuralFieldAgent] {

    override def update(agent: NeuralFieldAgent, others: Iterable[NeuralFieldAgent])(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {
      val directionsToOthersMat = DenseMatrix.zeros[Double](2, others.size)
      val distancesToOthersVec = DenseVector.zeros[Double](others.size)
      val egocentricNeuronDirectionsMat = DenseMatrix.zeros[Double](allocentricNeuronAngles.size, 2)

      val referenceVector =
        if (config.allocentricReferenceFrame) DenseVector[Double](1.0, 0.0) else agent.direction

      others.zipWithIndex.foreach({
        case (other, i) => {
          val vectorToOther = other.position - agent.position
          val distanceToOther = vectorToOther.norm()
          distancesToOthersVec(i) = distanceToOther

          val directionToOther =
            if (distanceToOther > 1e-9) vectorToOther / distanceToOther
            else {
              val randomAngle = config.random.nextDouble() * 2 * Pi
              DenseVector(cos(randomAngle), sin(randomAngle))
            }

          directionsToOthersMat(0, i) = directionToOther(0)
          directionsToOthersMat(1, i) = directionToOther(1)
        }
      })

      allocentricNeuronAngles.zipWithIndex.foreach({ case (angle, i) =>
        val rotatedNeuronDirection = rotateVector(referenceVector, angle)
        egocentricNeuronDirectionsMat(i, 0) = rotatedNeuronDirection(0)
        egocentricNeuronDirectionsMat(i, 1) = rotatedNeuronDirection(1)
      })

      val neuronTargetAngles = acos(
        max(min(egocentricNeuronDirectionsMat * directionsToOthersMat, 1.0), -1.0)
      )

      val externalStimuli =
        config.externalStimulusStrength * exp(
          -pow(-neuronTargetAngles, 2) /
            (2 * config.receptiveFieldVariance)
        )

      val neuronsExternalStimuli = sum(externalStimuli(*, ::))

      val nextMembranePotentials = calculateNextMembranePotentials(
        agent.membranePotentials,
        neuronsExternalStimuli
      )

      val activations =
        max(tanh(nextMembranePotentials * config.inverseTemperatureCoefficient), 0.0)

      val neuralForces = egocentricNeuronDirectionsMat(::, *) *:* activations

      val neuralForce = sum(neuralForces(::, *)).t

      val forceNorm = neuralForce.norm()

      if (forceNorm.isNaN || forceNorm < 1e-9)
        agent.copy(membranePotentials = nextMembranePotentials)
      else {
        val velocity = (config.averageSpeed / config.neuronsAmount) * neuralForce
        agent.copy(
          direction = velocity.normalize(),
          speed = velocity.norm(),
          membranePotentials = nextMembranePotentials
        )
      }

    }

    override def move(agent: NeuralFieldAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {
      val newPosition = agent.position + agent.direction * agent.speed * deltaTime

      agent.copy(position = newPosition)
    }

    override def translate(
        agent: NeuralFieldAgent,
        newPosition: DenseVector[Double]
    ): NeuralFieldAgent = {
      agent.copy(position = newPosition)
    }

    private def rotateVector(vector: DenseVector[Double], angle: Double): DenseVector[Double] =
      DenseVector[Double](
        cos(angle) * vector(0) - sin(angle) * vector(1),
        sin(angle) * vector(0) + cos(angle) * vector(1)
      )

    private def calculateNextMembranePotentials(
        membranePotentials: DenseVector[Double],
        externalStimuli: DenseVector[Double]
    )(implicit config: ParticleAgentConfig): DenseVector[Double] = {
      membranePotentials + config.timestepDuration * (
        -membranePotentials +
          (
            synapticConnectivityMatrix *
              tanh(membranePotentials * config.inverseTemperatureCoefficient)
          ) / config.neuronsAmount.toDouble -
          config.neuralInhibitionCoefficient +
          externalStimuli
      )

    }

  }
}
