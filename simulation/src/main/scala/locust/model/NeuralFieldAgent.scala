package pl.edu.agh.locust.model

import breeze.linalg.{DenseVector, DenseMatrix, Axis, *, norm, normalize, sum, min, max}
import breeze.numerics.{acos, cos, sin, atan2, sqrt, exp, pow, abs, tanh}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import pl.edu.agh.locust.utils.{Xorshift32, ParticleAgentUtils}
import scala.math.Pi
import breeze.linalg.operators.OpMulScalar

final case class NeuralFieldAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    id: Long,
    membranePotentials: DenseVector[Double],
    hopIterationsLeft: Int,
    activeTimeLeft: Double,
    isActive: Boolean
) extends ParticleAgent

object NeuralFieldAgent {
  private var allocentricNeuronAngles: Array[Double] = Array.empty
  private val rngState: ThreadLocal[Int] =
    ThreadLocal.withInitial(() => scala.util.Random.nextInt())

  private var synapticConnectivityMatrix = DenseMatrix.zeros[Double](1, 1)
  private var inverseTemperatureCoefficient: Double = 1.0

  def init()(implicit config: ParticleAgentConfig) = {
    inverseTemperatureCoefficient = config.inverseTemperatureCoefficient
    val neuronsAngleStep = (2 * Pi) / config.neuronsAmount
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep).toArray

    synapticConnectivityMatrix =
      DenseMatrix.tabulate(config.neuronsAmount, config.neuronsAmount)({ case (i, j) =>
        val angleDifference = abs(allocentricNeuronAngles(i) - allocentricNeuronAngles(j))
        val circularDistance = min(angleDifference, 2 * Pi - angleDifference)
        // Pi - abs(Pi - abs(allocentricNeuronAngles(i) - allocentricNeuronAngles(j)))
        cos(Pi * pow(circularDistance / Pi, config.synapticConnectivityCoefficient))
      })
  }

  def activations(agent: NeuralFieldAgent): Vector[Double] =
    max(tanh(agent.membranePotentials * inverseTemperatureCoefficient), 0.0).toArray.toVector

  def apply(
      position: DenseVector[Double],
      direction: DenseVector[Double],
      id: Long
  )(implicit config: ParticleAgentConfig): NeuralFieldAgent = {
    val headingAngle = atan2(direction(1), direction(0))
    val membranePotentials = new DenseVector(
      allocentricNeuronAngles.map(neuronAngle => {
        val angleDifference = abs(headingAngle - neuronAngle)
        val circularDistance = min(angleDifference, 2 * Pi - angleDifference)
        config.initialBumpAmplitude *
          exp(-circularDistance * circularDistance / (2 * config.receptiveFieldVariance)) +
          0.1 * config.random.nextGaussian()
      })
    )

    val initialActiveTimeLeft = config.activityPeriod -
      config.random.nextDouble() * (config.activityPeriod + config.minimalInactivityPeriod)

    NeuralFieldAgent(
      position,
      direction,
      id,
      membranePotentials,
      0,
      initialActiveTimeLeft,
      initialActiveTimeLeft > 0.0
    )
  }

  implicit case object Behaviour extends AgentBehaviour[NeuralFieldAgent] {

    override def update(agent: NeuralFieldAgent, others: Iterable[NeuralFieldAgent])(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {

      val (neuronsExternalStimuli, egocentricNeuronDirections, isRepulsionZoneOccupied) =
        ParticleAgentUtils.getNeuronsExternalStimuli(agent, others, 1.0)

      val nextMembranePotentials = calculateNextMembranePotentials(
        agent.membranePotentials,
        neuronsExternalStimuli
      )

      // (neurons)
      val activations: DenseVector[Double] =
        max(tanh(nextMembranePotentials * config.inverseTemperatureCoefficient), 0.0)

      val neuralForce = ParticleAgentUtils.getNeuralForce(activations, egocentricNeuronDirections)

      val forceNorm: Double = neuralForce.norm()

      val hopIterationsLeft =
        if (agent.hopIterationsLeft > 0) {
          agent.hopIterationsLeft - 1
        } else {
          val willHop =
            if (isRepulsionZoneOccupied)
              Xorshift32.nextFloat(rngState) < config.crowdedHopProbability
            else Xorshift32.nextFloat(rngState) < config.hopProbability

          if (willHop) {
            config.hopDurationTimesteps
          } else 0
        }

      val isActive = agent.activeTimeLeft > 0.0
      val inactiveTime = max(-agent.activeTimeLeft, 0.0)
      val reactivate =
        if (inactiveTime >= config.minimalInactivityPeriod)
          config.random.nextDouble() < config.resumeMarchProbabilityPerTimestep
        else false

      val activeTimeLeft =
        if (reactivate) config.activityPeriod else agent.activeTimeLeft - config.timestepDuration

      if (forceNorm.isNaN || forceNorm < 1e-9)
        agent.copy(
          membranePotentials = nextMembranePotentials,
          hopIterationsLeft = hopIterationsLeft,
          isActive = isActive,
          activeTimeLeft = activeTimeLeft
        )
      else
        agent.copy(
          direction = neuralForce.normalize(),
          membranePotentials = nextMembranePotentials,
          hopIterationsLeft = hopIterationsLeft,
          isActive = isActive,
          activeTimeLeft = activeTimeLeft
        )

    }

    override def move(agent: NeuralFieldAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {
      if (!agent.isActive) return agent
      val speed =
        if (agent.hopIterationsLeft <= 0) config.averageSpeed
        else config.hopSpeed

      val newPosition = agent.position + agent.direction * speed * deltaTime

      agent.copy(position = newPosition)
    }

    override def translate(
        agent: NeuralFieldAgent,
        newPosition: DenseVector[Double]
    ): NeuralFieldAgent = {
      agent.copy(position = newPosition)
    }

    override def getSpeed(agent: NeuralFieldAgent)(implicit config: ParticleAgentConfig): Double =
      if (!agent.isActive) 0.0
      else if (agent.hopIterationsLeft > 0) config.hopSpeed
      else config.averageSpeed

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
