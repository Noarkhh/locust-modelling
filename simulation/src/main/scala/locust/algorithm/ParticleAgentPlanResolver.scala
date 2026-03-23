package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.{PlanResolver, Update}
import pl.edu.agh.xinuk.model.{CellContents, Empty}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.xinuk.algorithm.Metrics
import pl.edu.agh.locust.algorithm.ParticleAgentMetrics

final case class ParticleAgentPlanResolver() extends PlanResolver[ParticleAgentConfig] {
  def isUpdateValid(iteration: Long, contents: CellContents, update: Update)(implicit
      config: ParticleAgentConfig
  ): Boolean = true

  def applyUpdate(iteration: Long, contents: CellContents, update: Update)(implicit
      config: ParticleAgentConfig
  ): (CellContents, Metrics) = {
    (contents, ParticleAgentMetrics.empty)
  }
}
