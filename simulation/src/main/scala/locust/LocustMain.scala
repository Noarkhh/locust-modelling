package pl.edu.agh.locust

import java.awt.Color

import com.typesafe.scalalogging.LazyLogging
import pl.edu.agh.locust.algorithm.{
  ParticleAgentMetrics
//   RabbitsPlanCreator,
//   RabbitsPlanResolver,
//   RabbitsWorldCreator
}
// import pl.edu.agh.rabbits.model.{Lettuce, Rabbit}
import pl.edu.agh.xinuk.Simulation
import pl.edu.agh.xinuk.model.CellState
import pl.edu.agh.xinuk.model.grid.GridSignalPropagation
import pl.edu.agh.xinuk.simulation.GridInfoCellColor
import pl.edu.agh.xinuk.algorithm.Metrics
import pl.edu.agh.xinuk.simulation.GridInfoCellParticles
import pl.edu.agh.locust.model.AgentContainer
import pl.edu.agh.locust.algorithm.ParticleAgentPlanCreator
import pl.edu.agh.locust.algorithm.ParticleAgentPlanResolver
import pl.edu.agh.locust.algorithm.ParticleAgentWorldCreator

object LocustMain extends LazyLogging {
  private val configPrefix = "particle-agent"

  def main(args: Array[String]): Unit = {
    import pl.edu.agh.xinuk.config.ValueReaders._
    new Simulation(
      configPrefix,
      ParticleAgentMetrics.MetricHeaders,
      ParticleAgentWorldCreator,
      ParticleAgentPlanCreator,
      ParticleAgentPlanResolver,
      ParticleAgentMetrics.empty,
      GridSignalPropagation.Standard,
      cellStatePayloader
    ).start()
  }

  private def cellStatePayloader(cellState: CellState): GridInfoCellParticles = {
    GridInfoCellParticles(cellState.contents match {
      case AgentContainer(agents, size) =>
        agents.map(agent => (agent.position(0) / size, agent.position(1) / size))
    })
  }

  // private def cellStatePayloader(cellState: CellState): GridInfoCellColor = {
  //   GridInfoCellColor(cellState.contents match {
  //     case _: Rabbit  => new Color(140, 69, 19)
  //     case _: Lettuce => new Color(0, 128, 0)
  //     case _          => Color.WHITE
  //   })
  // }
}
