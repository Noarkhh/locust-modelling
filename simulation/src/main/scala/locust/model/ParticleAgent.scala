package pl.edu.agh.locust.model

import java.awt.Color
import breeze.linalg.{DenseVector, norm, normalize}
import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.SPPAgentConfig

sealed trait Agent {
  val position: DenseVector[Double]
}

sealed trait AgentBehaviour[A <: Agent, C <: XinukConfig] {
  def update(agent: A, conspecifics: Set[A])(implicit config: C): A
  def move(agent: A, deltaTime: Double)(implicit config: C): A
  def translate(agent: A, newPosition: DenseVector[Double]): A
}

final case class SPPAgent(position: DenseVector[Double], direction: DenseVector[Double])
    extends Agent

object SPPAgent {
  implicit case object Behaviour extends AgentBehaviour[SPPAgent, SPPAgentConfig] {
    def update(agent: SPPAgent, others: Set[SPPAgent])(implicit
        config: SPPAgentConfig
    ): SPPAgent = {
      val socialForce: DenseVector[Double] = others
        .map(other => {
          val displacement: DenseVector[Double] = agent.position - other.position
          (norm(displacement), other.direction, normalize(displacement))
        })
        .toList
        .sortBy(_._1)
        .take(config.occlusionThreshold)
        .map({
          case (
                distance: Double,
                otherDirection: DenseVector[Double],
                displacementDirection: DenseVector[Double]
              ) => {
            if (distance < config.repulsionRange)
              -config.repulsionWeight * displacementDirection
            else if (distance < config.alignmentRange)
              config.alignmentWeight * otherDirection
            else if (distance < config.attractionRange)
              config.attractionWeight * displacementDirection
            else
              DenseVector.zeros[Double](2)
          }
        })
        .reduce(_ + _)

      val newDirection: DenseVector[Double] =
        normalize(
          config.previousDirectionWeight * agent.direction +
            (1 - config.previousDirectionWeight) * normalize(socialForce)
        )

      agent.copy(direction = newDirection)
    }

    def move(agent: SPPAgent, deltaTime: Double)(implicit config: SPPAgentConfig): SPPAgent = {
      val newPosition = agent.position + agent.direction * deltaTime * config.averageSpeed

      agent.copy(position = newPosition)
    }

    def translate(agent: SPPAgent, newPosition: DenseVector[Double]): SPPAgent = {
      agent.copy(position = newPosition)
    }
  }
}

final case class AgentContainer[A <: Agent](
    var agents: Set[A],
    var lastUpdateIteration: Long,
    behaviour: AgentBehaviour[A, SPPAgentConfig],
    size: Double,
    xMin: Double,
    yMin: Double,
    particlesColor: Color
) extends CellContents
