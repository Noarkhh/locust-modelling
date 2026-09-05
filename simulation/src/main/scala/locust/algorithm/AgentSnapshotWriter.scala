package pl.edu.agh.locust.algorithm

import java.io.{BufferedOutputStream, FileOutputStream, OutputStream}
import java.lang.management.ManagementFactory
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Paths}

import breeze.numerics.atan2

import pl.edu.agh.locust.config.ParticleAgentConfig
import pl.edu.agh.locust.model.{AgentBehaviour, NeuralFieldAgent, ParticleAgent, SPPAgent}

/** Appends fixed-width little-endian agent records to a binary file, one file
  * per JVM. All workers of a node share the writer; rows from different cells
  * interleave freely and are grouped by the iteration field when read.
  *
  * Record layout (25 bytes), numpy dtype:
  *   [('iter','<u4'),('id','<u4'),('x','<f4'),('y','<f4'),
  *    ('heading','<f4'),('speed','<f4'),('flags','u1')]
  * flags: bit 0 = active, bit 1 = hopping.
  */
object AgentSnapshotWriter {
  val RecordSize = 25

  private var out: Option[OutputStream] = None
  private var initialized = false

  def shouldSnapshot(iteration: Long)(implicit config: ParticleAgentConfig): Boolean =
    config.snapshotPath.nonEmpty &&
      iteration >= config.snapshotStartIteration &&
      (iteration - config.snapshotStartIteration) % config.snapshotFrequency == 0

  def append[A <: ParticleAgent](
      iteration: Long,
      agents: Iterable[A],
      behaviour: AgentBehaviour[A]
  )(implicit config: ParticleAgentConfig): Unit = {
    if (agents.isEmpty) return
    val buffer = ByteBuffer.allocate(agents.size * RecordSize).order(ByteOrder.LITTLE_ENDIAN)
    agents.foreach { agent =>
      val flags = agent match {
        case nfa: NeuralFieldAgent =>
          (if (nfa.isActive) 1 else 0) | (if (nfa.hopIterationsLeft > 0) 2 else 0)
        case spp: SPPAgent =>
          (if (spp.isActive) 1 else 0) | (if (spp.hopIterationsLeft > 0) 2 else 0)
        case _ => 1
      }
      buffer.putInt(iteration.toInt)
      buffer.putInt(agent.id.toInt)
      buffer.putFloat(agent.position(0).toFloat)
      buffer.putFloat(agent.position(1).toFloat)
      buffer.putFloat(atan2(agent.direction(1), agent.direction(0)).toFloat)
      buffer.putFloat(behaviour.getSpeed(agent).toFloat)
      buffer.put(flags.toByte)
    }
    synchronized {
      get().foreach(_.write(buffer.array()))
    }
  }

  def close(): Unit = synchronized {
    out.foreach(_.close())
    out = None
    initialized = false
  }

  private def get()(implicit config: ParticleAgentConfig): Option[OutputStream] = {
    if (!initialized) {
      initialized = true
      val dir = Paths.get(config.snapshotPath)
      Files.createDirectories(dir)
      val runtimeName = ManagementFactory.getRuntimeMXBean.getName.replace('@', '-')
      val file = dir.resolve(s"snapshots-$runtimeName.bin").toFile
      out = Some(new BufferedOutputStream(new FileOutputStream(file), 1 << 16))
    }
    out
  }
}
