// xinuk-core is brought in as a source subproject of this build (not a separate
// build via ProjectRef) so it gets full LSP support and is editable in place.
// Its settings are ported from deps/xinuk/build.sbt. The xinuk example projects
// (rabbits, torch, urban, mock, fortwist) are intentionally excluded.
lazy val xinukCore = (project in file("deps/xinuk/xinuk-core"))
  .settings(
    name := "xinuk-core",
    organization := "pl.edu.agh",
    scalaVersion := "2.13.18",
    scalacOptions ++= Seq(
      "-feature",
      "-deprecation",
      "-unchecked",
      "-language:implicitConversions",
      "-language:existentials",
      "-language:dynamics",
      "-language:experimental.macros",
      "-language:higherKinds",
      "-Xlint:-missing-interpolator,-adapted-args,-unused,_"
    ),
    libraryDependencies ++= Seq(
      "com.avsystem.commons" %% "commons-core" % "2.0.0-M12",
      "io.altoo" %% "akka-kryo-serialization" % "1.0.0",
      "com.iheart" %% "ficus" % "1.5.0",
      "com.typesafe.akka" %% "akka-actor" % "2.6.13",
      "com.typesafe.akka" %% "akka-slf4j" % "2.6.13",
      "com.typesafe.akka" %% "akka-cluster" % "2.6.13",
      "com.typesafe.akka" %% "akka-cluster-sharding" % "2.6.13",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2",
      "org.scala-lang.modules" %% "scala-swing" % "2.1.1",
      "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.11.2",
      "org.jfree" % "jfreechart" % "1.5.0",
      "org.scalatest" %% "scalatest" % "3.2.2" % Test,
      "com.typesafe.akka" %% "akka-testkit" % "2.6.13" % Test,
      "org.mockito" % "mockito-core" % "3.5.10" % Test
    )
  )

lazy val quadTree = project in file("deps/quadtree-scala")

lazy val locustSimulation = (project in file("."))
  .settings(
    name := "locust-simulation",
    organization := "pl.edu.agh",
    version := "0.1.0",
    scalaVersion := "2.13.18",
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
