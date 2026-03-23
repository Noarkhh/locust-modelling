package pl.edu.afg.locust

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

object LocustMain extends LazyLogging {
  private val configPrefix = "locust"

  def main(args: Array[String]): Unit = {
    import pl.edu.agh.xinuk.config.ValueReaders._
    // new Simulation(
    //   configPrefix,
    //   ParticleAgentMetrics.MetricHeaders,
    //   RabbitsWorldCreator,
    //   RabbitsPlanCreator,
    //   RabbitsPlanResolver,
    //   RabbitsMetrics.empty,
    //   GridSignalPropagation.Standard,
    //   cellStatePayloader
    // ).start()
  }

  // private def cellStatePayloader(cellState: CellState): GridInfoCellColor = {
  //   GridInfoCellColor(cellState.contents match {
  //     case _: Rabbit  => new Color(140, 69, 19)
  //     case _: Lettuce => new Color(0, 128, 0)
  //     case _          => Color.WHITE
  //   })
  // }
}
