package pl.edu.agh.locust.algorithm

import scala.math.Pi

import pl.edu.agh.xinuk.algorithm.WorldCreator
import pl.edu.agh.xinuk.model.{CellContents, CellState, WorldBuilder}
import pl.edu.agh.xinuk.model.grid.{GridCellId, GridWorldBuilder}
import pl.edu.agh.locust.config.SPPAgentConfig
import pl.edu.agh.locust.model.AgentContainer
import pl.edu.agh.locust.model.SPPAgent
import pl.edu.agh.locust.model.SPPAgent.Behaviour
import breeze.linalg.{normalize, norm}
import breeze.linalg.DenseVector
import breeze.numerics.cos
import breeze.numerics.sin
import java.awt.Color

object ParticleAgentWorldCreator extends WorldCreator[SPPAgentConfig] {

  override def prepareWorld()(implicit config: SPPAgentConfig): WorldBuilder = {
    val worldBuilder = GridWorldBuilder().withGridConnections().withWrappedBoundaries()

    val colors = Map(
      (0, 0) -> Color.BLACK,
      (0, 1) -> Color.MAGENTA,
      (0, 2) -> Color.BLUE,
      (1, 0) -> Color.GREEN,
      (1, 1) -> Color.RED,
      (1, 2) -> Color.ORANGE,
      (2, 0) -> Color.PINK,
      (2, 1) -> Color.YELLOW,
      (2, 2) -> Color.DARK_GRAY
    )

    for {
      x <- 0 until config.worldWidth
      y <- 0 until config.worldHeight
    } {
      val agentsAmount =
        (config.meanAgentDensity * config.agentContainerSize * config.agentContainerSize).toInt
      var agents = Range(0, agentsAmount)
        .map(i => {

          val position =
            DenseVector(
              (x + config.random.nextDouble()) * config.agentContainerSize,
              (y + config.random.nextDouble()) * config.agentContainerSize
            )

          val angle = config.random.nextDouble() * 2 * Pi
          val direction = DenseVector(cos(angle), sin(angle))

          SPPAgent(position, direction)
        })
        .toSet

      val contents = AgentContainer(
        agents,
        -1,
        SPPAgent.Behaviour,
        config.agentContainerSize,
        x * config.agentContainerSize,
        y * config.agentContainerSize,
        colors.getOrElse((x, y), Color.BLACK)
      )
      worldBuilder(GridCellId(x, y)) = CellState(contents)
    }

    worldBuilder
  }
}
