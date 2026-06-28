package pl.edu.agh.locust.model

import breeze.linalg.{DenseVector, norm, normalize, sum, min, max}
import breeze.numerics.{acos, cos, sin, atan2}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import scala.math.Pi
import breeze.numerics.sqrt
import breeze.numerics.exp
import breeze.numerics.pow
import breeze.numerics.abs

final case class SpinSystemAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    id: Long,
    speed: Double,
    var neuronSpinStates: Array[Int]
) extends ParticleAgent

object SpinSystemAgent {
  var allocentricNeuronAngles: Seq[Double] = Seq.empty
  var synapticConnectivity: Seq[Double] = Seq.empty

  def init()(implicit config: ParticleAgentConfig) = {
    val neuronsAngleStep = (2 * Pi) / config.neuronsAmount
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep)
    allocentricNeuronAngles = Range(0, config.neuronsAmount).map(_ * neuronsAngleStep)
    synapticConnectivity = (0.0 +: allocentricNeuronAngles.tail
      .map(neuronAngle =>
        cos(Pi * pow(neuronAngle / Pi, config.synapticConnectivityCoefficient))
      )).toSeq
    // synapticConnectivity.foreach(i => print(s" $i"))
    // println()
  }

  def apply(
      position: DenseVector[Double],
      direction: DenseVector[Double],
      id: Long
  )(implicit config: ParticleAgentConfig): SpinSystemAgent = {
    val neuronSpinStates = Array.fill(config.neuronsAmount) {
      if (config.random.nextBoolean()) 1 else -1
    }
    SpinSystemAgent(position, direction, id, config.averageSpeed, neuronSpinStates)
  }

  implicit case object Behaviour extends AgentBehaviour[SpinSystemAgent] {

    override def update(agent: SpinSystemAgent, others: Iterable[SpinSystemAgent])(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      // val referenceAngle =
      //   if (config.allocentricReferenceFrame) 0.0 else atan2

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
                // if (agent.id == 0) println(neuronTargetAngle)

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
        .zip(agent.neuronSpinStates)
        .filter({ case (direction, spin) => spin > 0 })
        .map({ case (direction, spin) => direction })
        .reduce(_ + _)

      // if (agent.id == 0) {
      //   agent.neuronSpinStates.foreach(i => print(s" $i"))
      //   println()
      // }

      val velocity = (config.averageSpeed / config.neuronsAmount) * neuralForce

      agent.copy(direction = velocity.normalize(), speed = velocity.norm())

    }

    override def move(agent: SpinSystemAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): SpinSystemAgent = {
      // if (agent.id == 0) println(agent.speed)
      // if (agent.id == 0) println(agent.direction)

      val newPosition = agent.position + agent.direction * agent.speed * deltaTime

      agent.copy(position = newPosition)
    }

    override def translate(
        agent: SpinSystemAgent,
        newPosition: DenseVector[Double]
    ): SpinSystemAgent = {
      agent.copy(position = newPosition)
    }

    private def rotateVector(vector: DenseVector[Double], angle: Double): DenseVector[Double] =
      DenseVector[Double](
        cos(angle) * vector(0) - sin(angle) * vector(1),
        sin(angle) * vector(0) + cos(angle) * vector(1)
      )

    private def getNeuronPairConnectivity(neuron1Index: Int, neuron2Index: Int): Double = {
      val neuronsGap = min(
        abs(neuron1Index - neuron2Index),
        synapticConnectivity.size - abs(neuron1Index - neuron2Index)
      )
      synapticConnectivity(neuronsGap)
    }

    private def runNeuralDynamics(
        neuronSpinStates: Array[Int],
        externalStimuli: Seq[Double],
        agentId: Long
    )(implicit config: ParticleAgentConfig) = {
      for (iteration <- 0 until config.neuralDynamicIterationsPerTimestep) {
        val neuronToFlip = config.random.nextInt(config.neuronsAmount)

        // val hamiltonianBefore = calculateHamiltonian(neuronSpinStates, externalStimuli)
        // neuronSpinStates(neuronToFlip) *= -1
        // val hamiltonianAfter = calculateHamiltonian(neuronSpinStates, externalStimuli)
        // neuronSpinStates(neuronToFlip) *= -1
        // val deltaHamiltonian = hamiltonianAfter - hamiltonianBefore
        val deltaHamiltonian =
          calculateDeltaHamiltonian(neuronToFlip, neuronSpinStates, externalStimuli)
        val flipProbability = exp(
          -config.inverseTemperatureCoefficient * deltaHamiltonian
        )
        // if (agentId == 0 && iteration == 0) {
        //   println(f"externalStimuli: ")
        //   externalStimuli.foreach(i => print(f" $i%.3f"))
        //   println()
        // println(f"H1 = $hamiltonianBefore%.5f H2 = $hamiltonianAfter%.5f")
        //   println(f"dH = $deltaHamiltonian%.3f")
        //   println(f"p(flip) = $flipProbability%.3f")
        // }
        if (config.random.nextDouble() < flipProbability) neuronSpinStates(neuronToFlip) *= -1
      }
    }

    private def calculateDeltaHamiltonian(
        selectedNeuron: Int,
        neuronSpinStates: Array[Int],
        externalStimuli: Seq[Double]
    )(implicit config: ParticleAgentConfig): Double = {
      (2 * neuronSpinStates(selectedNeuron)) *
        (
          neuronSpinStates.zipWithIndex
            .map({ case (jSpinState, j) =>
              getNeuronPairConnectivity(selectedNeuron, j) * jSpinState
            })
            .sum / config.neuronsAmount +
            externalStimuli(selectedNeuron) -
            config.neuralInhibitionCoefficient
        )
    }

    private def calculateHamiltonian(
        neuronSpinStates: Array[Int],
        externalStimuli: Seq[Double]
    )(implicit config: ParticleAgentConfig): Double = {
      -neuronSpinStates.zipWithIndex
        .map({ case (iSpinState, i) =>
          val connectivityFactor =
            neuronSpinStates.zipWithIndex
              .map({ case (jSpinState, j) =>
                jSpinState * iSpinState * getNeuronPairConnectivity(i, j)
              })
              .sum / config.neuronsAmount

          val externalStimuliFactor =
            iSpinState * (externalStimuli(i) - config.neuralInhibitionCoefficient)

          connectivityFactor + externalStimuliFactor
        })
        .sum
    }
  }
}
