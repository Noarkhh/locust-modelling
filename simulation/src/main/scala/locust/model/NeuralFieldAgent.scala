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
        val angleBetweenNeurons =
          Pi - abs(Pi - abs(allocentricNeuronAngles(i) - allocentricNeuronAngles(j)))
        cos(Pi * pow(angleBetweenNeurons / Pi, config.synapticConnectivityCoefficient))
      })
  }

  def activations(agent: NeuralFieldAgent): Vector[Double] =
    max(tanh(agent.membranePotentials * inverseTemperatureCoefficient), 0.0).toArray.toVector

  def apply(
      position: DenseVector[Double],
      direction: DenseVector[Double],
      id: Long
  )(implicit config: ParticleAgentConfig): NeuralFieldAgent = {
    // Ring state encoding the agent's initial heading: a Gaussian bump of
    // configurable amplitude at the heading angle (width = receptive field
    // sigma), plus small noise. Amplitude 0 recovers the paper's rest-state
    // init (bump forms freely at the social consensus — near-instant
    // ordering); a formed bump must instead be ROTATED by social input, so
    // ordering proceeds on the bump-rotation timescale.
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
    // Stagger the initial march/pause timer uniformly over the full cycle
    // (Bach 2018) so the population does not march and pause in lockstep.
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
      val referenceVector: DenseVector[Double] =
        if (config.allocentricReferenceFrame) DenseVector[Double](1.0, 0.0) else agent.direction

      // (2, others)
      val directionsToGoals = DenseMatrix.zeros[Double](2, others.size)
      // (others)
      val distancesToOthers = DenseVector.zeros[Double](others.size)
      // (others)
      val egocentricAnglesToOthers = DenseVector.zeros[Double](others.size)

      // (others)
      val isOtherPursuing = DenseVector.zeros[Boolean](others.size)

      var isRepulsionZoneOccupied = false

      others.zipWithIndex.foreach({
        case (other, i) => {
          val vectorToOther = other.position - agent.position
          val distanceToOther = vectorToOther.norm()
          distancesToOthers(i) = distanceToOther

          val directionToOther =
            if (distanceToOther > 1e-9) vectorToOther / distanceToOther
            else {
              val randomAngle = config.random.nextDouble() * 2 * Pi
              DenseVector(cos(randomAngle), sin(randomAngle))
            }

          val egocentricAngleToOther =
            acos(max(min(agent.direction dot directionToOther, 1.0), -1.0))

          egocentricAnglesToOthers(i) = egocentricAngleToOther

          // True pursuit test: the other's heading points AT this agent
          // (angle between the other's direction and the bearing other->agent),
          // not merely parallel to this agent's own heading.
          val otherAngleToAgent =
            acos(max(min(other.direction dot (-directionToOther), 1.0), -1.0))

          if (
            (egocentricAngleToOther > config.antiGoalAngleRangeStart) &&
            (otherAngleToAgent < config.pursuerHeadingAngleEnd) &&
            (distanceToOther < config.antiGoalOverrideRange)
          ) {
            directionsToGoals(0, i) = -directionToOther(0)
            directionsToGoals(1, i) = -directionToOther(1)
            isOtherPursuing(i) = true

            if (distanceToOther < config.repulsionRange) isRepulsionZoneOccupied = true
          } else {
            directionsToGoals(0, i) = directionToOther(0)
            directionsToGoals(1, i) = directionToOther(1)
            isOtherPursuing(i) = false
          }

        }
      })
      // if (agent.id == 0) println(egocentricAnglesToOthers)

      // (neurons, 2)
      val egocentricNeuronDirections = DenseMatrix.zeros[Double](allocentricNeuronAngles.size, 2)
      allocentricNeuronAngles.zipWithIndex.foreach({ case (angle, i) =>
        val rotatedNeuronDirection = rotateVector(referenceVector, angle)
        egocentricNeuronDirections(i, 0) = rotatedNeuronDirection(0)
        egocentricNeuronDirections(i, 1) = rotatedNeuronDirection(1)
      })

      // (neurons, others)
      val neuronTargetAngles: DenseMatrix[Double] = acos(
        max(min(egocentricNeuronDirections * directionsToGoals, 1.0), -1.0)
      )

      // val externalStimuliStrengths = DenseVector.fill(others.size) {
      //   config.externalStimulusStrength
      // }
      // externalStimuliStrengths(distancesToOthers <:< config.antiGoalOverrideRange) =
      //   config.antiGoalStimulusStrength

      // (others)
      val externalStimuliStrengths: DenseVector[Double] = DenseVector.tabulate(others.size) { i =>
        {
          // val angleToOther = acos()

          if (isOtherPursuing(i)) {
            // if (agent.id == 0) println("Aaa")
            config.antiGoalStimulusStrength
          } else config.totalSocialAttraction / others.size
        }
      }

      // (neurons, others)
      val receptiveFieldResponse: DenseMatrix[Double] =
        exp(
          -pow(-neuronTargetAngles, 2) /
            (2 * config.receptiveFieldVariance)
        )

      // (neurons, others)
      val externalStimuli: DenseMatrix[Double] =
        receptiveFieldResponse(*, ::) *:* externalStimuliStrengths

      // (neurons)
      val neuronsExternalStimuli: DenseVector[Double] = sum(externalStimuli(*, ::))

      // (neurons)
      val nextMembranePotentials = calculateNextMembranePotentials(
        agent.membranePotentials,
        neuronsExternalStimuli
      )

      // (neurons)
      val activations: DenseVector[Double] =
        max(tanh(nextMembranePotentials * config.inverseTemperatureCoefficient), 0.0)

      // (neurons, 2)
      val neuralForces: DenseMatrix[Double] = egocentricNeuronDirections(::, *) *:* activations

      // (2)
      val neuralForce: DenseVector[Double] = sum(neuralForces(::, *)).t

      // ()
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
