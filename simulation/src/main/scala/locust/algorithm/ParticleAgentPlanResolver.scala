package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.{PlanResolver, Update}
import pl.edu.agh.xinuk.model.{CellContents, Empty}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.xinuk.algorithm.Metrics
import pl.edu.agh.locust.algorithm.ParticleAgentMetrics
import pl.edu.agh.locust.algorithm.ParticleAgentUpdate._
import pl.edu.agh.locust.model.{Agent, AgentContainer, SPPAgent}
import java.awt.Color

final case class ParticleAgentPlanResolver() extends PlanResolver[ParticleAgentConfig] {
  def isUpdateValid(iteration: Long, contents: CellContents, update: Update)(implicit
      config: ParticleAgentConfig
  ): Boolean = true

  def applyUpdate(iteration: Long, contents: CellContents, update: Update)(implicit
      config: ParticleAgentConfig
  ): (CellContents, Metrics) = {
    val result = update match {
      case a: AddAgents[_] => applyContainerUpdate(iteration, contents, a)
    }
    (result, ParticleAgentMetrics.empty)
  }

  private def applyContainerUpdate[A <: Agent](
      iteration: Long,
      contents: CellContents,
      addAgents: AddAgents[A]
  ): AgentContainer[A] = {
    val agentContainer = contents.asInstanceOf[AgentContainer[A]]
    if (iteration > agentContainer.lastUpdateIteration) {
      agentContainer.lastUpdateIteration = iteration
      agentContainer.agents = addAgents.agents
    } else {
      agentContainer.agents ++= addAgents.agents
    }
    agentContainer
  }
}
