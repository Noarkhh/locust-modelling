package pl.edu.afg.locust.model

import breeze.linalg.DenseVector
import pl.edu.agh.xinuk.model.{CellContents, Signal}

trait Agent {
  val position: DenseVector[Float]
  val velocity: DenseVector[Float]

  def update(conspecifics: Set[Agent]): Agent

  def move(): Agent
}

final case class SPPAgent(position: DenseVector[Float], velocity: DenseVector[Float])
    extends Agent {
  override def update(conspecifics: Set[Agent]): Agent = {
    this
  }

  override def move(): Agent = {
    this
  }
}
final case object SPPAgent

final case class NeuralFieldAgent(x: Long, y: Long)
final case class SpinSystemAgent(x: Long, y: Long)

final case class AgentContainer(var agents: Set[Agent]) extends CellContents
