# Repilot Patch

```
 // TODO: remove this in 2.0
```

# Developer Patch

```

```

# Context

```
--- bug/Closure-31/src/com/google/javascript/jscomp/Compiler.java

+++ fix/Closure-31/src/com/google/javascript/jscomp/Compiler.java

@@ -1282,7 +1282,7 @@

 
       // Check if the sources need to be re-ordered.
       if (options.dependencyOptions.needsManagement() &&
-          !options.skipAllPasses &&
+ // TODO: remove this in 2.0
           options.closurePass) {
         for (CompilerInput input : inputs) {
           // Forward-declare all the provided types, so that they
```

# Note

