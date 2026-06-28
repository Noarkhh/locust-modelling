package pl.edu.agh.locust.utils
import breeze.linalg.{DenseVector, norm, normalize, sum}
import java.lang.Float.intBitsToFloat
import breeze.numerics.abs

object ImplicitVectorOps {
  implicit class DenseVectorOps(v: DenseVector[Double]) {
    def normalize(): DenseVector[Double] = breeze.linalg.normalize(v)
    def norm(): Double = breeze.linalg.norm(v)
  }
}

object Xorshift32 {
  def next(state: Int): Int = {
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^ x << 5
  }

  def next(state: ThreadLocal[Int]): Int = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    next
  }

  def nextInt(state: ThreadLocal[Int], n: Int): Int = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    abs(next) % n
  }

  def nextFloat(state: ThreadLocal[Int]): Float = {
    val next = Xorshift32.next(state.get())
    state.set(next)
    intBitsToFloat((next >>> 9) | 0x3f800000) - 1.0f
  }
}
