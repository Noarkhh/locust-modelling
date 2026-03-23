package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.WorldCreator
import pl.edu.agh.xinuk.model.{CellContents, CellState, WorldBuilder}
import pl.edu.agh.xinuk.model.grid.{GridCellId, GridWorldBuilder}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.model.AgentContainer
import breeze.linalg.DenseVector
import pl.edu.agh.locust.model.SPPAgent

object ParticleAgentWorldCreator extends WorldCreator[ParticleAgentConfig] {

  override def prepareWorld()(implicit config: ParticleAgentConfig): WorldBuilder = {
    val worldBuilder = GridWorldBuilder().withGridConnections()

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
              config.random.nextDouble() * config.agentContainerSize,
              config.random.nextDouble() * config.agentContainerSize
            )

          SPPAgent(position, DenseVector(0.0, 0.0))
        })
        .toSet

      val contents = AgentContainer(agents, config.agentContainerSize)
      // val contents: CellContents = if (config.random.nextDouble() < config.rabbitSpawnChance) {
      //   Rabbit(config.rabbitStartEnergy, 0)
      // } else {
      //   Lettuce(0)
      // }

      worldBuilder(GridCellId(x, y)) = CellState(contents)
    }

    worldBuilder
  }
}
