# Repilot Patch

```
if (options.dependencyOptions.needsManagement()) {
```

# Developer Patch

```
      if (options.dependencyOptions.needsManagement()) {
```

# Context

```
--- bug/Closure-18/src/com/google/javascript/jscomp/Compiler.java

+++ fix/Closure-18/src/com/google/javascript/jscomp/Compiler.java

@@ -1285,7 +1285,7 @@

 
       // Check if the sources need to be re-ordered.
       boolean staleInputs = false;
-      if (options.dependencyOptions.needsManagement() && options.closurePass) {
+if (options.dependencyOptions.needsManagement()) {
         for (CompilerInput input : inputs) {
           // Forward-declare all the provided types, so that they
           // are not flagged even if they are dropped from the process.
```

# Note

