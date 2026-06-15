package pl.edu.agh.locust.model

import scala.math.Pi
import java.awt.Color
import breeze.linalg.{DenseVector, norm, normalize, sum, min}
import breeze.numerics.{cos, sin}
import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import scala.collection.mutable.PriorityQueue
import breeze.linalg.max

sealed trait Agent {
  val position: DenseVector[Double]
  val direction: DenseVector[Double]
}

sealed trait AgentBehaviour[A <: Agent, C <: XinukConfig] {
  def update(agent: A, others: Iterable[A])(implicit config: C): A
  def move(agent: A, deltaTime: Double)(implicit config: C): A
  def translate(agent: A, newPosition: DenseVector[Double]): A
}

final case class SPPAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    activeTimeLeft: Double,
    isActive: Boolean,
    nextIterationDirection: DenseVector[Double],
    hopIterationsLeft: Int,
    id: Long
) extends Agent

object SPPAgent {
  def apply(position: DenseVector[Double], direction: DenseVector[Double], id: Long)(implicit
      config: ParticleAgentConfig
  ): SPPAgent =
    SPPAgent(
      position,
      direction,
      config.activityPeriod,
      true,
      direction,
      0,
      id
    )

  implicit case object Behaviour extends AgentBehaviour[SPPAgent, ParticleAgentConfig] {
    def update(agent: SPPAgent, others: Iterable[SPPAgent])(implicit
        config: ParticleAgentConfig
    ): SPPAgent = {
      if (others.size == 0) return agent

      val (socialForce, isRepulsionZoneOccupied) = calculateSocialForce(agent, others)

      val randomAngle = config.random.nextDouble() * 2 * Pi
      val randomDirection = DenseVector(cos(randomAngle), sin(randomAngle))

      val noisedDirection =
        (config.randomComponentWeight * randomDirection + (1 - config.randomComponentWeight) * agent.direction)
          .normalize()

      val newDirection: DenseVector[Double] =
        (
          config.previousDirectionWeight * noisedDirection +
            (1 - config.previousDirectionWeight) * socialForce.normalize()
        ).normalize()

      val hopIterationsLeft =
        if (agent.hopIterationsLeft > 0) {
          agent.hopIterationsLeft - 1
        } else {
          val willHop =
            if (isRepulsionZoneOccupied) config.random.nextDouble() < config.crowdedHopProbability
            else config.random.nextDouble() < config.hopProbability

          if (willHop) config.hopDurationTimesteps
          else 0
        }

      val isActive = agent.activeTimeLeft > 0.0
      val inactiveTime = max(-agent.activeTimeLeft, 0.0)
      val reactivate =
        if (inactiveTime >= config.minimalInactivityPeriod)
          config.random.nextDouble() < config.resumeMarchProbabilityPerTimestep
        else false

      val activeTimeLeft =
        if (reactivate) config.activityPeriod else agent.activeTimeLeft - config.timestepDuration

      agent.copy(
        nextIterationDirection = newDirection,
        hopIterationsLeft = hopIterationsLeft,
        isActive = isActive,
        activeTimeLeft = activeTimeLeft
      )
    }

    def move(agent: SPPAgent, deltaTime: Double)(implicit config: ParticleAgentConfig): SPPAgent = {
      if (!agent.isActive) return agent
      val speed =
        if (agent.hopIterationsLeft <= 0) config.averageSpeed
        else config.hopSpeed

      val newPosition = agent.position + agent.nextIterationDirection * deltaTime * speed

      agent.copy(position = newPosition, direction = agent.nextIterationDirection)
    }

    def translate(agent: SPPAgent, newPosition: DenseVector[Double]): SPPAgent = {
      agent.copy(position = newPosition)
    }

    private def calculateSocialForce(
        agent: SPPAgent,
        others: Iterable[SPPAgent]
    )(implicit config: ParticleAgentConfig): (DenseVector[Double], Boolean) =
      others
        .map(other => {
          val displacement = agent.position - other.position
          val distance = displacement.norm()
          val displacementDirection = displacement.normalize()

          val socialForceFactor =
            if (distance < config.repulsionRange)
              config.repulsionWeight * displacementDirection
            else if (distance < config.alignmentRange)
              config.alignmentWeight * other.direction
            else if (distance < config.attractionRange)
              -config.attractionWeight * displacementDirection
            else
              DenseVector.zeros[Double](2)

          (socialForceFactor, distance < config.repulsionRange)
        })
        .reduce(((a1, a2) => (a1._1 + a2._1, a1._2 || a2._2)))

  }
}

final case class AgentContainer[A <: Agent](
    var agents: Iterable[A],
    var lastUpdateIteration: Long,
    behaviour: AgentBehaviour[A, ParticleAgentConfig],
    size: Double,
    xMin: Double,
    yMin: Double
) extends CellContents
