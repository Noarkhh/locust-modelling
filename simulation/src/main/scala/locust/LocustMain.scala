package pl.edu.agh.locust

import java.awt.Color

import com.typesafe.scalalogging.LazyLogging
import pl.edu.agh.xinuk.Simulation
import pl.edu.agh.xinuk.model.CellState
import pl.edu.agh.xinuk.model.grid.GridSignalPropagation
import pl.edu.agh.xinuk.simulation.GuiCellColor
import pl.edu.agh.xinuk.algorithm.Metrics
import pl.edu.agh.xinuk.simulation.{GuiCellParticles, GuiParticle}

import pl.edu.agh.locust.model.{ParticleAgent, AgentContainer}
import pl.edu.agh.locust.algorithm.{
  ParticleAgentMetrics,
  ParticleAgentPlanCreator,
  ParticleAgentPlanResolver,
  ParticleAgentWorldCreator
}
import pl.edu.agh.locust.model.SPPAgent
import scala.math.Pi

import breeze.numerics.atan2
import breeze.numerics.acos
import breeze.linalg.DenseVector
import pl.edu.agh.locust.utils.ImplicitVectorOps.DenseVectorOps

object LocustMain extends LazyLogging {
  private val configPrefix = "particle-agent"

  def main(args: Array[String]): Unit = {
    import pl.edu.agh.xinuk.config.ValueReaders._
    import pl.edu.agh.locust.config.ValueReaders._
    // println(atan2(0, 1) * 180 / Pi)
    // println(atan2(1, 1) * 180 / Pi)
    // println(atan2(1, 0) * 180 / Pi)
    // println(atan2(1, -1) * 180 / Pi)
    // println(atan2(0, -1) * 180 / Pi)
    // println(atan2(-1, -1) * 180 / Pi)
    // println(atan2(-1, 0) * 180 / Pi)
    // println(atan2(-1, 1) * 180 / Pi)
    //
    // val vec1 = DenseVector[Double](1.0, 0.0).normalize()
    // val vec2 = DenseVector[Double](1.0, -1.0).normalize()

    // println(acos(vec1.dot(vec2)) * 180 / Pi)

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

  private def cellStatePayloader(cellState: CellState): GuiCellParticles = {

    val container = cellState.contents.asInstanceOf[AgentContainer[ParticleAgent]]

    val particles = container.agents.map(agent => {
      if (
        container.xMin > agent.position(0) || agent.position(0) > container.xMin + container.size
      ) {
        println("agent out of container")
      }

      GuiParticle(
        (agent.position(0) - container.xMin) / container.size,
        (agent.position(1) - container.yMin) / container.size
      )
    })

    GuiCellParticles(particles)
  }

  // private def cellStatePayloader(cellState: CellState): GuiCellColor = {
  //   GuiCellColor(cellState.contents match {
  //     case _: Rabbit  => new Color(140, 69, 19)
  //     case _: Lettuce => new Color(0, 128, 0)
  //     case _          => Color.WHITE
  //   })
  // }
}
