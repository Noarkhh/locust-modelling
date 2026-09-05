package pl.edu.agh.locust.config

import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.xinuk.config.GuiType
import pl.edu.agh.xinuk.model.WorldType
import scala.util.Random
import java.security.SecureRandom
import breeze.numerics.pow
import scala.math.Pi
import com.avsystem.commons.misc.NamedEnum
import com.avsystem.commons.misc.AbstractNamedEnumCompanion
import net.ceedubs.ficus.readers.ValueReader
import com.typesafe.config.Config
import pl.edu.agh.locust.model.ParticleAgent
import breeze.linalg.DenseVector
import pl.edu.agh.locust.model.{SPPAgent, SpinSystemAgent}
import pl.edu.agh.locust.model.AgentBehaviour
import pl.edu.agh.locust.model.NeuralFieldAgent

final case class ParticleAgentConfig(
    // Generic xinuk config
    worldType: WorldType,
    worldWidthMeters: Double,
    worldHeightMeters: Double,
    guiHeight: Int,
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
    guiStartIteration: Long,
    guiUpdateFrequency: Long,
    // Parameter search instrumentation
    snapshotPath: String,
    snapshotFrequency: Long,
    snapshotStartIteration: Long,
    randomSeed: Long,
    // Particle agnet factory
    particleAgentFactory: ParticleAgentFactory,
    // Particle agent metrics config
    localOrderNeighbours: Int,
    // Particle agent config
    timestepDuration: Double,
    agentContainerSize: Double,
    agentAmount: Int,
    initialHeadingSpread: Double,
    initialAreaCenterX: Double,
    initialAreaCenterY: Double,
    initialAreaRadiusX: Double,
    initialAreaRadiusY: Double,
    averageSpeed: Double,
    // SPP agent config
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
    resumeMarchProbabilityPerSecond: Double,
    // Ring Attractor agent config
    neuronsAmount: Int,
    synapticConnectivityCoefficient: Double,
    inverseTemperatureCoefficient: Double,
    neuralInhibitionCoefficient: Double,
    totalSocialAttraction: Double,
    allocentricReferenceFrame: Boolean,
    // Spin system agent config
    neuralDynamicIterationsPerNeuronPerSecond: Int,
    // Neural Field agent config
    antiGoalOverrideRange: Double,
    antiGoalStimulusStrength: Double,
    receptiveFieldStd: Double,
    antiGoalAngleRangeStart: Double,
    pursuerHeadingAngleEnd: Double,
    initialBumpAmplitude: Double
) extends XinukConfig {
  val worldWidth: Int = (worldWidthMeters / agentContainerSize).toInt
  val worldHeight: Int = (worldHeightMeters / agentContainerSize).toInt
  val guiCellSize: Int = (guiHeight / worldHeight).toInt
  val random: Random = if (randomSeed != 0) new Random(randomSeed) else new SecureRandom
  val hopDurationTimesteps: Int = (hopDuration / timestepDuration).toInt
  val resumeMarchProbabilityPerTimestep: Double =
    1.0 - pow((1.0 - resumeMarchProbabilityPerSecond), timestepDuration)
  val neuralDynamicIterationsPerTimestep: Int =
    (neuralDynamicIterationsPerNeuronPerSecond * timestepDuration * neuronsAmount).toInt
  // val receptiveFieldVariance = pow((spatialDiscretizationCoefficient * 2 * Pi) / neuronsAmount, 2)
  val receptiveFieldVariance = pow(receptiveFieldStd, 2)
  val externalStimulusStrength = totalSocialAttraction / occlusionThreshold
}

sealed trait ParticleAgentFactory extends NamedEnum {
  def instantiateAgent(position: DenseVector[Double], direction: DenseVector[Double], id: Long)(
      implicit config: ParticleAgentConfig
  ): ParticleAgent
  def getAgentBehaviour(): AgentBehaviour[ParticleAgent]
  def initAgentCompanion()(implicit
      config: ParticleAgentConfig
  ): Unit = ()

}

// ParticleAgentFactory[SPPAgent] <: ParticleAgentFactory[ParticleAgent]
// ParticleAgentSerializer[ParticleAgent] <: ParticleAgentSerializer[SPPAgent]

// trait ParticleAgentSerializer[-A <: ParticleAgent] {
//   def serializeAgent(agent: A): String
// }

object ParticleAgentFactory extends AbstractNamedEnumCompanion[ParticleAgentFactory] {

  override val values: List[ParticleAgentFactory] = caseObjects

  case object SPPAgentFactory extends ParticleAgentFactory {
    override val name: String = "SPPAgentFactory"
    override def instantiateAgent(
        position: DenseVector[Double],
        direction: DenseVector[Double],
        id: Long
    )(implicit config: ParticleAgentConfig): ParticleAgent = {
      SPPAgent(position, direction, id)
    }

    override def getAgentBehaviour(): AgentBehaviour[ParticleAgent] =
      SPPAgent.Behaviour.asInstanceOf[AgentBehaviour[ParticleAgent]]
  }

  case object SpinSystemAgentFactory extends ParticleAgentFactory {
    override def name: String = "SpinSystemAgentFactory"
    override def instantiateAgent(
        position: DenseVector[Double],
        direction: DenseVector[Double],
        id: Long
    )(implicit config: ParticleAgentConfig): ParticleAgent = {
      SpinSystemAgent(position, direction, id)
    }
    override def getAgentBehaviour(): AgentBehaviour[ParticleAgent] =
      SpinSystemAgent.Behaviour.asInstanceOf[AgentBehaviour[ParticleAgent]]
    override def initAgentCompanion()(implicit config: ParticleAgentConfig): Unit =
      SpinSystemAgent.init()
  }

  case object NeuralFieldAgentFactory extends ParticleAgentFactory {
    override def name: String = "NeuralFieldAgentFactory"
    override def instantiateAgent(
        position: DenseVector[Double],
        direction: DenseVector[Double],
        id: Long
    )(implicit config: ParticleAgentConfig): ParticleAgent = {
      NeuralFieldAgent(position, direction, id)
    }
    override def getAgentBehaviour(): AgentBehaviour[ParticleAgent] =
      NeuralFieldAgent.Behaviour.asInstanceOf[AgentBehaviour[ParticleAgent]]
    override def initAgentCompanion()(implicit config: ParticleAgentConfig): Unit =
      NeuralFieldAgent.init()
  }
}

object ValueReaders {
  implicit val agentTypeReader: ValueReader[ParticleAgentFactory] =
    new ValueReader[ParticleAgentFactory] {
      override def read(config: Config, path: String): ParticleAgentFactory =
        ParticleAgentFactory.byName(config.getString(path))
    }
}
