# Repilot Patch

```
          inputs != null &&
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
+          inputs != null &&
           options.closurePass) {
         for (CompilerInput input : inputs) {
           // Forward-declare all the provided types, so that they
```

# Note

`inputs` can not be `null` as it's used letter