package pl.edu.agh.locust.config

import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.xinuk.config.GuiType
import pl.edu.agh.xinuk.model.WorldType
import scala.util.Random
import java.security.SecureRandom

final case class ParticleAgentConfig(
    worldType: WorldType,
    worldWidth: Int,
    worldHeight: Int,
    iterationsNumber: Long,
    iterationFinishedLogFrequency: Long,
    skipEmptyLogs: Boolean,
    signalSuppressionFactor: Double,
    signalAttenuationFactor: Double,
    signalDisabled: Boolean,
    workersX: Int,
    workersY: Int,
    isSupervisor: Boolean,
    shardingMod: Int,
    guiType: GuiType,
    guiCellSize: Int,
    guiStartIteration: Long,
    guiUpdateFrequency: Long,
    agentContainerSize: Double,
    meanAgentDensity: Double
) extends XinukConfig {
  val random: Random = new SecureRandom
}
