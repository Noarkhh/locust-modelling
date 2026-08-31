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
    membranePotentials: DenseVector[Double],
    hopIterationsLeft: Int
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
    // val membranePotentials =
    //   DenseVector.tabulate(config.neuronsAmount)(i => 0.1 * config.random.nextGaussian())
    val membranePotentials = new DenseVector(
      allocentricNeuronAngles.map(neuronAngle => {
        val headingAngle = atan2(direction(1), direction(0))
        val angleBias = abs(Pi - abs(headingAngle - neuronAngle)) / Pi
        config.random.nextGaussian() * 0.05 + angleBias * 0.95
        // config.random.nextGaussian() * 0.5 + angleBias * 0.5
        // config.random.nextGaussian() * 0.75 + angleBias * 0.25
      })
    )
    NeuralFieldAgent(position, direction, id, config.averageSpeed, membranePotentials, 0)
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

          val otherAngleToAgent =
            acos(max(min(agent.direction dot other.direction, 1.0), -1.0))

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

      if (agent.id == 0) println(activations)

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

      if (forceNorm.isNaN || forceNorm < 1e-9)
        agent.copy(
          membranePotentials = nextMembranePotentials,
          hopIterationsLeft = hopIterationsLeft
        )
      else {
        val velocity: DenseVector[Double] =
          (config.averageSpeed / config.neuronsAmount) * neuralForce
        if (agent.id == 0) println(velocity.norm())
        if (agent.id == 0) println(velocity)
        agent.copy(
          direction = velocity.normalize(),
          speed = velocity.norm(),
          membranePotentials = nextMembranePotentials,
          hopIterationsLeft = hopIterationsLeft
        )
      }

    }

    override def move(agent: NeuralFieldAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {
      val speed =
        if (agent.hopIterationsLeft <= 0) agent.speed
        else {
          config.hopSpeed
        }

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
      agent.speed

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
