package pl.edu.agh.locust.config

import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.xinuk.config.GuiType
import pl.edu.agh.xinuk.model.WorldType
import scala.util.Random
import java.security.SecureRandom
import breeze.numerics.pow

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
    guiParticleSize: Int,
    guiStartIteration: Long,
    guiUpdateFrequency: Long,
    timestepDuration: Double,
    agentContainerSize: Double,
    agentAmount: Int,
    initialAreaCenterX: Double,
    initialAreaCenterY: Double,
    initialAreaRadius: Double,
    averageSpeed: Double,
    previousDirectionWeight: Double,
    randomComponentWeight: Double,
    repulsionRange: Double,
    alignmentRange: Double,
    attractionRange: Double,
    repulsionWeight: Double,
    alignmentWeight: Double,
    attractionWeight: Double,
    occlusionThreshold: Int,
    hopProbability: Double,
    crowdedHopProbability: Double,
    hopDuration: Double,
    hopSpeed: Double,
    activityPeriod: Double,
    minimalInactivityPeriod: Double,
    resumeMarchProbabilityPerSec: Double,
    localOrderNeighbours: Int
) extends XinukConfig {
  val random: Random = new SecureRandom
  val hopDurationTimesteps: Int = (hopDuration / timestepDuration).toInt
  val resumeMarchProbabilityPerTimestep: Double =
    1.0 - pow((1.0 - resumeMarchProbabilityPerSec), timestepDuration)
  println(resumeMarchProbabilityPerTimestep)
}
