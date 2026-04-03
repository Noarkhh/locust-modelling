package pl.edu.agh.locust.config

import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.xinuk.config.GuiType
import pl.edu.agh.xinuk.model.WorldType
import scala.util.Random
import java.security.SecureRandom

final case class SPPAgentConfig(
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
    guiParticleSize: Int,
    guiCellSize: Int,
    guiStartIteration: Long,
    guiUpdateFrequency: Long,
    timestepLength: Double,
    agentContainerSize: Double,
    meanAgentDensity: Double,
    averageSpeed: Double,
    previousDirectionWeight: Double,
    randomComponentWeight: Double,
    repulsionRange: Double,
    alignmentRange: Double,
    attractionRange: Double,
    repulsionWeight: Double,
    alignmentWeight: Double,
    attractionWeight: Double,
    occlusionThreshold: Int
) extends XinukConfig {
  val random: Random = new SecureRandom
}
