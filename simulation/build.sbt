lazy val xinukCore = ProjectRef(file("deps/xinuk"), "xinuk-core")
lazy val quadTree = ProjectRef(file("deps/quadtree-scala"), "quadtree-scala")

lazy val locustSimulation = (project in file("."))
  .settings(
    name := "locust-simulation",
    organization := "pl.edu.agh",
    version := "0.1.0",
    scalaVersion := "2.13.17",
    Compile / mainClass := Some("pl.edu.agh.locust.LocustMain"),
    run / fork := true,
    javaOptions += "--add-modules=jdk.incubator.vector",
    libraryDependencies ++= Seq(
      "org.scalanlp" %% "breeze" % "2.1.0",
      "ch.qos.logback" % "logback-classic" % "1.2.3"
    )
  )
  .settings(
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf"              => MergeStrategy.concat
      case x                             => MergeStrategy.first
    }
  )
  .dependsOn(xinukCore, quadTree)
