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
    // Particle agnet factory
    particleAgentFactory: ParticleAgentFactory,
    // Particle agent metrics config
    localOrderNeighbours: Int,
    // Particle agent config
    timestepDuration: Double,
    agentContainerSize: Double,
    agentAmount: Int,
    initialAreaCenterX: Double,
    initialAreaCenterY: Double,
    initialAreaRadius: Double,
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
    // Spin system agent config
    neuronsAmount: Int,
    neuralDynamicIterationsPerNeuronPerSecond: Int,
    spatialDiscretizationCoefficient: Double,
    synapticConnectivityCoefficient: Double,
    inverseTemperatureCoefficient: Double,
    neuralInhibitionCoefficient: Double,
    totalSocialAttraction: Double,
    allocentricReferenceFrame: Boolean
) extends XinukConfig {
  val random: Random = new SecureRandom
  val hopDurationTimesteps: Int = (hopDuration / timestepDuration).toInt
  val resumeMarchProbabilityPerTimestep: Double =
    1.0 - pow((1.0 - resumeMarchProbabilityPerSecond), timestepDuration)
  val neuralDynamicIterationsPerTimestep: Int =
    (neuralDynamicIterationsPerNeuronPerSecond * timestepDuration * neuronsAmount).toInt
  val receptiveFieldVariance = pow((spatialDiscretizationCoefficient * 2 * Pi) / neuronsAmount, 2)
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
