package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.Update
import pl.edu.agh.locust.model.Agent

object ParticleAgentUpdate {
  case class AddAgents[A <: Agent](agents: Set[A]) extends Update
}
