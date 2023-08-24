# Repilot Patch

```
int indexOfDot = namespace.lastIndexOf('.');
```

# Developer Patch

```
          int indexOfDot = namespace.lastIndexOf('.');
```

# Context

```
--- bug/Closure-92/src/com/google/javascript/jscomp/ProcessClosurePrimitives.java

+++ fix/Closure-92/src/com/google/javascript/jscomp/ProcessClosurePrimitives.java

@@ -786,7 +786,7 @@

         } else {
           // In this case, the name was implicitly provided by two independent
           // modules. We need to move this code up to a common module.
-          int indexOfDot = namespace.indexOf('.');
+int indexOfDot = namespace.lastIndexOf('.');
           if (indexOfDot == -1) {
             // Any old place is fine.
             compiler.getNodeForCodeInsertion(minimumModule)
```

# Note

