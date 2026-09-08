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
import pl.edu.agh.locust.utils.ParticleAgentUtils

final case class SpinSystemAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    id: Long,
    var neuronSpinStates: DenseVector[Double],
    hopIterationsLeft: Int,
    activeTimeLeft: Double,
    isActive: Boolean
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
      if (config.initialBumpAmplitude > 0.0) {
        val headingAngle = atan2(direction(1), direction(0))
        new DenseVector[Double](
          allocentricNeuronAngles
            .map(neuronAngle => {
              val angleDifference = abs(headingAngle - neuronAngle)
              val circularDistance = min(angleDifference, 2 * Pi - angleDifference)

              if (
                cos(Pi * pow(circularDistance / Pi, config.synapticConnectivityCoefficient)) >= 0.0
              )
                1.0
              else -1.0
            })
            .toArray
        )
      } else
        DenseVector.tabulate(config.neuronsAmount)(_ =>
          if (config.random.nextBoolean()) 1.0 else -1.0
        )
    val initialActiveTimeLeft = config.activityPeriod -
      config.random.nextDouble() * (config.activityPeriod + config.minimalInactivityPeriod)
    SpinSystemAgent(
      position,
      direction,
      id,
      neuronSpinStates,
      0,
      initialActiveTimeLeft,
      initialActiveTimeLeft > 0.0
    )
  }

  implicit case object Behaviour extends AgentBehaviour[SpinSystemAgent] {

    override def update(agent: SpinSystemAgent, others: Iterable[SpinSystemAgent])(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      val externalStimulusMultiplier = 1 / sqrt(2 * Pi * config.receptiveFieldVariance)

      val (neuronsExternalStimuli, egocentricNeuronDirections, isRepulsionZoneOccupied) =
        ParticleAgentUtils.getNeuronsExternalStimuli(agent, others, externalStimulusMultiplier)

      runNeuralDynamics(agent.neuronSpinStates, neuronsExternalStimuli)

      val activeSpins = max(agent.neuronSpinStates, 0.0)
      val neuralForce = ParticleAgentUtils.getNeuralForce(activeSpins, egocentricNeuronDirections)

      val forceNorm = neuralForce.norm()

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
          hopIterationsLeft = hopIterationsLeft,
          isActive = isActive,
          activeTimeLeft = activeTimeLeft
        )
      else
        agent.copy(
          direction = neuralForce.normalize(),
          hopIterationsLeft = hopIterationsLeft,
          isActive = isActive,
          activeTimeLeft = activeTimeLeft
        )
    }

    override def move(agent: SpinSystemAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      if (!agent.isActive) return agent
      val speed =
        if (agent.hopIterationsLeft <= 0) config.averageSpeed
        else config.hopSpeed

      val newPosition = agent.position + agent.direction * speed * deltaTime

      agent.copy(position = newPosition)
    }

    override def translate(
        agent: SpinSystemAgent,
        newPosition: DenseVector[Double]
    ): SpinSystemAgent = {
      agent.copy(position = newPosition)
    }

    override def getSpeed(agent: SpinSystemAgent)(implicit config: ParticleAgentConfig): Double =
      if (!agent.isActive) 0.0
      else if (agent.hopIterationsLeft > 0) config.hopSpeed
      else config.averageSpeed

    private def runNeuralDynamics(
        neuronSpinStates: DenseVector[Double],
        externalStimuli: DenseVector[Double]
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
        externalStimuli: DenseVector[Double]
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
