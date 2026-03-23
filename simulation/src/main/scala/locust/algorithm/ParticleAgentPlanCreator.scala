package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.PlanCreator
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.xinuk.algorithm.{Metrics, Plans}
import pl.edu.agh.xinuk.model.{CellContents, CellId, CellState, Direction}

final case class ParticleAgentPlanCreator() extends PlanCreator[ParticleAgentConfig] {
  def createPlans(
      iteration: Long,
      cellId: CellId,
      cellState: CellState,
      neighbourContents: Map[Direction, CellContents]
  )(implicit config: ParticleAgentConfig): (Plans, Metrics) = {
    (Plans.empty, ParticleAgentMetrics.empty)
  }
}
