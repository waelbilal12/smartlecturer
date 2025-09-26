// File: android/build.gradle.kts

// تم تغيير هذا القسم بالكامل ليناسب صيغة Kotlin DSL الصحيحة
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        // تم تصحيح طريقة إضافة الـ classpath هنا
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.23")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}