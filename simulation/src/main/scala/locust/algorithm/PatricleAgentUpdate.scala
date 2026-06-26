package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.Update
import pl.edu.agh.locust.model.ParticleAgent

object ParticleAgentUpdate {
  case class AddAgents[A <: ParticleAgent](agents: Iterable[A]) extends Update
}
