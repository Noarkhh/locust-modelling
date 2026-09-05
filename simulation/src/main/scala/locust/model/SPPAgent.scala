package pl.edu.agh.locust.model

import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.utils.ImplicitVectorOps._
import pl.edu.agh.locust.model.{ParticleAgent, AgentBehaviour}
import breeze.linalg.{DenseVector, norm, normalize, sum, min, max}
import breeze.numerics.{cos, sin}
import scala.math.Pi

final case class SPPAgent(
    position: DenseVector[Double],
    direction: DenseVector[Double],
    activeTimeLeft: Double,
    isActive: Boolean,
    nextIterationDirection: DenseVector[Double],
    hopIterationsLeft: Int,
    id: Long
) extends ParticleAgent

object SPPAgent {
  def apply(position: DenseVector[Double], direction: DenseVector[Double], id: Long)(implicit
      config: ParticleAgentConfig
  ): SPPAgent = {
    // Stagger the initial march/pause timer uniformly over the full cycle
    // (Bach 2018) so the population does not march and pause in lockstep.
    val initialActiveTimeLeft = config.activityPeriod -
      config.random.nextDouble() * (config.activityPeriod + config.minimalInactivityPeriod)
    SPPAgent(
      position,
      direction,
      initialActiveTimeLeft,
      initialActiveTimeLeft > 0.0,
      direction,
      0,
      id
    )
  }

  implicit case object Behaviour extends AgentBehaviour[SPPAgent] {
    override def update(agent: SPPAgent, others: Iterable[SPPAgent])(implicit
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

    override def move(agent: SPPAgent, deltaTime: Double)(implicit
        config: ParticleAgentConfig
    ): SPPAgent = {
      if (!agent.isActive) return agent
      val speed =
        if (agent.hopIterationsLeft <= 0) config.averageSpeed
        else config.hopSpeed

      val newPosition = agent.position + agent.nextIterationDirection * deltaTime * speed

      agent.copy(position = newPosition, direction = agent.nextIterationDirection)
    }

    override def translate(agent: SPPAgent, newPosition: DenseVector[Double]): SPPAgent = {
      agent.copy(position = newPosition)
    }

    override def getSpeed(agent: SPPAgent)(implicit config: ParticleAgentConfig): Double =
      if (!agent.isActive) 0.0
      else if (agent.hopIterationsLeft > 0) config.averageSpeed + config.hopSpeed
      else config.averageSpeed

    private def calculateSocialForce(
        agent: SPPAgent,
        others: Iterable[SPPAgent]
    )(implicit config: ParticleAgentConfig): (DenseVector[Double], Boolean) =
      others
        .map(other => {
          val vectorToOther = other.position - agent.position
          val distanceToOther = vectorToOther.norm()
          val directionToOther = vectorToOther.normalize()

          val socialForceFactor =
            if (distanceToOther < config.repulsionRange)
              -config.repulsionWeight * directionToOther
            else if (distanceToOther < config.alignmentRange)
              config.alignmentWeight * other.direction
            else if (distanceToOther < config.attractionRange)
              config.attractionWeight * directionToOther
            else
              DenseVector.zeros[Double](2)

          (socialForceFactor, distanceToOther < config.repulsionRange)
        })
        .reduce(((a1, a2) => (a1._1 + a2._1, a1._2 || a2._2)))

  }
}
