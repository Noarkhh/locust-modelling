package pl.edu.agh.locust.utils
import breeze.linalg.{DenseVector, DenseMatrix, *, norm, normalize, sum, min, max}
import breeze.numerics.{acos, cos, sin, atan2, sqrt, exp, pow, abs, tanh}
import java.lang.Float.intBitsToFloat
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import breeze.numerics.abs
import pl.edu.agh.locust.model.ParticleAgent
import pl.edu.agh.locust.config.ParticleAgentConfig
import scala.math.Pi

object ImplicitVectorOps {
  implicit class DenseVectorOps(v: DenseVector[Double]) {
    def normalize(): DenseVector[Double] = breeze.linalg.normalize(v)
    def norm(): Double = breeze.linalg.norm(v)
  }
}

object Xorshift32 {
  def next(state: Int): Int = {
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^ x << 5
  }

  def next(state: ThreadLocal[Int]): Int = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    next
  }

  def nextInt(state: ThreadLocal[Int], n: Int): Int = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    abs(next) % n
  }

  def nextFloat(state: ThreadLocal[Int]): Float = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    intBitsToFloat((next >>> 9) | 0x3f800000) - 1.0f
  }
}

object ParticleAgentUtils {
  private var allocentricNeuronAngles: Array[Double] = Array.empty

  def init()(implicit config: ParticleAgentConfig) = {
    val neuronsAngleStep = (2 * Pi) / config.neuronsAmount
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep).toArray
  }

  def getNeuronsExternalStimuli[A <: ParticleAgent](
      agent: A,
      others: Iterable[A],
      externalStimulusMultiplier: Double
  )(implicit
      config: ParticleAgentConfig
  ): (DenseVector[Double], DenseMatrix[Double], Boolean) = {

    val referenceVector: DenseVector[Double] =
      if (config.allocentricReferenceFrame) DenseVector[Double](1.0, 0.0) else agent.direction

    // (2, others)
    val directionsToGoals = DenseMatrix.zeros[Double](2, others.size)

    // (others)
    val externalStimuliStrengths = DenseVector.zeros[Double](others.size)

    var isRepulsionZoneOccupied = false

    others.zipWithIndex.foreach({
      case (other, i) => {
        val vectorToOther = other.position - agent.position
        val distanceToOther = vectorToOther.norm()

        val directionToOther =
          if (distanceToOther > 1e-9) vectorToOther / distanceToOther
          else {
            val randomAngle = config.random.nextDouble() * 2 * Pi
            DenseVector(cos(randomAngle), sin(randomAngle))
          }

        val egocentricAngleToOther =
          acos(max(min(agent.direction dot directionToOther, 1.0), -1.0))

        val otherAngleToAgent =
          acos(max(min(other.direction dot (-directionToOther), 1.0), -1.0))

        if (
          (egocentricAngleToOther > config.antiGoalAngleRangeStart) &&
          (otherAngleToAgent < config.pursuerHeadingAngleEnd) &&
          (distanceToOther < config.antiGoalOverrideRange)
        ) {
          directionsToGoals(0, i) = -directionToOther(0)
          directionsToGoals(1, i) = -directionToOther(1)
          externalStimuliStrengths(i) = config.antiGoalStimulusStrength

          if (distanceToOther < config.repulsionRange) isRepulsionZoneOccupied = true
        } else {
          directionsToGoals(0, i) = directionToOther(0)
          directionsToGoals(1, i) = directionToOther(1)
          externalStimuliStrengths(i) =
            config.totalSocialAttraction * externalStimulusMultiplier / others.size
        }

      }
    })

    // (neurons, 2)
    val egocentricNeuronDirections = DenseMatrix.zeros[Double](config.neuronsAmount, 2)

    allocentricNeuronAngles.zipWithIndex.foreach({ case (angle, i) =>
      val rotatedNeuronDirection = rotateVector(referenceVector, angle)
      egocentricNeuronDirections(i, 0) = rotatedNeuronDirection(0)
      egocentricNeuronDirections(i, 1) = rotatedNeuronDirection(1)
    })

    // (neurons, others)
    val neuronTargetAngles: DenseMatrix[Double] = acos(
      max(min(egocentricNeuronDirections * directionsToGoals, 1.0), -1.0)
    )

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

    (neuronsExternalStimuli, egocentricNeuronDirections, isRepulsionZoneOccupied)

  }

  def getNeuralForce(
      neuronActivations: DenseVector[Double],
      egocentricNeuronDirections: DenseMatrix[Double]
  )(implicit config: ParticleAgentConfig): DenseVector[Double] = {
    // (neurons, 2)
    val neuralForces: DenseMatrix[Double] = egocentricNeuronDirections(::, *) *:* neuronActivations

    // (2)
    val neuralForce: DenseVector[Double] = sum(neuralForces(::, *)).t

    neuralForce
  }

  private def rotateVector(vector: DenseVector[Double], angle: Double): DenseVector[Double] =
    DenseVector[Double](
      cos(angle) * vector(0) - sin(angle) * vector(1),
      sin(angle) * vector(0) + cos(angle) * vector(1)
    )
}
