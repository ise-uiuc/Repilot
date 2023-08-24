# Repilot Patch

```
    if (isAssignmentTarget(n)) {
      return n;
    }
```

# Developer Patch

```
    if (isAssignmentTarget(n)) {
      return n;
    }
```

# Context

```
--- bug/Closure-161/src/com/google/javascript/jscomp/PeepholeFoldConstants.java

+++ fix/Closure-161/src/com/google/javascript/jscomp/PeepholeFoldConstants.java

@@ -1280,6 +1280,9 @@

     // If GETPROP/GETELEM is used as assignment target the array literal is
     // acting as a temporary we can't fold it here:
     //    "[][0] += 1"
+    if (isAssignmentTarget(n)) {
+      return n;
+    }
 
     if (right.getType() != Token.NUMBER) {
       // Sometimes people like to use complex expressions to index into
```

# Note

