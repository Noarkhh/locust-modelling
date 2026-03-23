package pl.agh.edu.pl.locust.algorithm

import pl.edu.agh.xinuk.algorithm.WorldCreator
import pl.edu.agh.xinuk.model.{CellContents, CellState, WorldBuilder}
import pl.edu.agh.xinuk.model.grid.{GridCellId, GridWorldBuilder}
import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.afg.locust.model.AgentContainer

object RabbitsWorldCreator extends WorldCreator[ParticleAgentConfig] {

  override def prepareWorld()(implicit config: ParticleAgentConfig): WorldBuilder = {
    val worldBuilder = GridWorldBuilder().withGridConnections()

    for {
      x <- 0 until config.worldWidth
      y <- 0 until config.worldHeight
    } {
      val contents = AgentContainer(Set.empty)
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
