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
import breeze.linalg.operators.OpMulScalar
import breeze.numerics.tanh

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
        // config.random.nextGaussian() +
        angleBias
      })
    )
    NeuralFieldAgent(position, direction, id, config.averageSpeed, membranePotentials)
  }

  implicit case object Behaviour extends AgentBehaviour[NeuralFieldAgent] {

    override def update(agent: NeuralFieldAgent, others: Iterable[NeuralFieldAgent])(implicit
        config: ParticleAgentConfig
    ): NeuralFieldAgent = {
      val targets: Iterable[(DenseVector[Double], Double)] = others.flatMap(other => {
        val vectorToOther = other.position - agent.position
        val distanceToOther = vectorToOther.norm()
        if (distanceToOther < 1e-10) None
        else Some((vectorToOther / distanceToOther, distanceToOther))
      })

      val referenceVector =
        if (config.allocentricReferenceFrame) DenseVector[Double](1.0, 0.0) else agent.direction

      val egocentricNeuronDirections = allocentricNeuronAngles.map(rotateVector(referenceVector, _))

      val neuronsExternalStimuli =
        egocentricNeuronDirections
          .map(egocentricNeuronDirection => {
            targets
              .map({ case (directionToOther, distanceToOther) =>
                val neuronTargetAngle =
                  acos(min(1.0, max(-1.0, egocentricNeuronDirection.dot(directionToOther))))

                (config.externalStimulusStrength) * exp(
                  -pow(-neuronTargetAngle, 2) /
                    (2 * config.receptiveFieldVariance)
                )
              })
              .sum
          })
          .toArray

      val nextMembranePotentials = calculateNextMembranePotentials(
        agent.membranePotentials,
        new DenseVector(neuronsExternalStimuli)
      )

      val neuralForce = egocentricNeuronDirections
        .zip(nextMembranePotentials.toArray)
        .map({ case (direction, potential) =>
          direction * max(0.0, tanh(potential * config.inverseTemperatureCoefficient))
        })
        .reduce(_ + _)

      val forceNorm = neuralForce.norm()

      if (forceNorm < 1e-9)
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

    // private def activation[T](
    //     potential: T
    // )(implicit
    //     config: ParticleAgentConfig,
    //     mul: OpMulScalar.Impl2[T, Double, T],
    //     impl: tanh.Impl[T, T]
    // ): T = {
    //   tanh(mul(potential, config.inverseTemperatureCoefficient))
    // }
    //
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
