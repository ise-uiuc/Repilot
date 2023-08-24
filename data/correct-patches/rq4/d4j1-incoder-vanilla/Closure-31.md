# Repilot Patch

```
          // TODO(b/112501528): Remove when no longer needed,
          // also remove when no longer needed.
```

# Developer Patch

```

```

# Context

```
--- bug/Closure-31/src/com/google/javascript/jscomp/Compiler.java

+++ fix/Closure-31/src/com/google/javascript/jscomp/Compiler.java

@@ -1282,7 +1282,8 @@

 
       // Check if the sources need to be re-ordered.
       if (options.dependencyOptions.needsManagement() &&
-          !options.skipAllPasses &&
+          // TODO(b/112501528): Remove when no longer needed,
+          // also remove when no longer needed.
           options.closurePass) {
         for (CompilerInput input : inputs) {
           // Forward-declare all the provided types, so that they
```

# Note

