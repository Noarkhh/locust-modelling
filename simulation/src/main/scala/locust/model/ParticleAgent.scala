package pl.edu.agh.locust.model

import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.ParticleAgentConfig
import breeze.linalg.DenseVector

trait ParticleAgent {
  val position: DenseVector[Double]
  val direction: DenseVector[Double]
  val id: Long
}

trait AgentBehaviour[A <: ParticleAgent] {
  def update(agent: A, others: Iterable[A])(implicit config: ParticleAgentConfig): A
  def move(agent: A, deltaTime: Double)(implicit config: ParticleAgentConfig): A
  def translate(agent: A, newPosition: DenseVector[Double]): A
  def getSpeed(agent: A)(implicit config: ParticleAgentConfig): Double
}

final case class AgentContainer[A <: ParticleAgent](
    var agents: Iterable[A],
    var lastUpdateIteration: Long,
    behaviour: AgentBehaviour[A],
    size: Double,
    xMin: Double,
    yMin: Double
) extends CellContents
