# Repilot Patch

```

          if (gramps.isDelProp()) {
            return false;
          }
```

# Developer Patch

```
          if (gramps.isDelProp()) {
            return false;
          }
```

# Context

```
--- bug/Closure-5/src/com/google/javascript/jscomp/InlineObjectLiterals.java

+++ fix/Closure-5/src/com/google/javascript/jscomp/InlineObjectLiterals.java

@@ -173,6 +173,10 @@

 
           // Deleting a property has different semantics from deleting
           // a variable, so deleted properties should not be inlined.
+
+          if (gramps.isDelProp()) {
+            return false;
+          }
 
           // NOTE(nicksantos): This pass's object-splitting algorithm has
           // a blind spot. It assumes that if a property isn't defined on an
```

# Note

