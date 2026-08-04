package pl.edu.agh.locust.algorithm

import scala.math.Pi

import pl.edu.agh.xinuk.algorithm.WorldCreator
import pl.edu.agh.xinuk.model.{CellContents, CellState, WorldBuilder}
import pl.edu.agh.xinuk.model.grid.{GridCellId, GridWorldBuilder}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.model.AgentContainer
import breeze.linalg.{normalize, norm}
import breeze.linalg.DenseVector
import breeze.numerics.cos
import breeze.numerics.sin
import java.awt.Color
import breeze.numerics.sqrt

object ParticleAgentWorldCreator extends WorldCreator[ParticleAgentConfig] {

  override def prepareWorld()(implicit config: ParticleAgentConfig): WorldBuilder = {
    val worldBuilder = GridWorldBuilder().withGridConnections().withWrappedBoundaries()
    config.particleAgentFactory.initAgentCompanion()
    val verticalScaleFactor = config.initialAreaRadiusY / config.initialAreaRadiusX

    val agents = List
      .tabulate(config.agentAmount)(i => {
        val r = config.initialAreaRadiusX * sqrt(config.random.nextDouble())
        val theta = config.random.nextDouble() * 2 * Pi

        val agentX = config.initialAreaCenterX + r * cos(theta)
        val agentY = config.initialAreaCenterY + r * sin(theta) * verticalScaleFactor
        val agentPosition = DenseVector(agentX, agentY)

        val noiseAngle = (config.random.nextDouble() * 2 * Pi) - Pi
        val agentAngle = noiseAngle * 0.25
        val agentDirection = DenseVector(cos(agentAngle), sin(agentAngle))
        config.particleAgentFactory.instantiateAgent(agentPosition, agentDirection, i)
      })
      .groupBy(agent => {
        (
          (agent.position(0) / config.agentContainerSize).toInt,
          (agent.position(1) / config.agentContainerSize).toInt
        )
      })

    for {
      x <- 0 until config.worldWidth
      y <- 0 until config.worldHeight
    } {
      // val agentsAmount =
      //   // if (x == 0 && y == 0)
      //   (config.meanAgentDensity * config.agentContainerSize * config.agentContainerSize).toInt
      // // else 0
      // var agents = Range(0, agentsAmount)
      //   .map(i => {
      //
      //     val position =
      //       DenseVector(
      //         (x + config.random.nextDouble()) * config.agentContainerSize,
      //         (y + config.random.nextDouble()) * config.agentContainerSize
      //       )
      //
      //     val angle = config.random.nextDouble() * 2 * Pi
      //     val direction = DenseVector(cos(angle), sin(angle))
      //
      //     SPPAgent(position, direction)
      //   })
      //   .toSet

      val contents = AgentContainer(
        agents.getOrElse((x, y), List.empty),
        -1,
        config.particleAgentFactory.getAgentBehaviour(),
        config.agentContainerSize,
        x * config.agentContainerSize,
        y * config.agentContainerSize
      )
      worldBuilder(GridCellId(x, y)) = CellState(contents)
    }

    worldBuilder
  }
}
