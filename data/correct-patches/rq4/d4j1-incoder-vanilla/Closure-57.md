# Repilot Patch

```
          if (target != null
              && target.getType() == Token.STRING) {
```

# Developer Patch

```
          if (target != null && target.getType() == Token.STRING) {
```

# Context

```
--- bug/Closure-57/src/com/google/javascript/jscomp/ClosureCodingConvention.java

+++ fix/Closure-57/src/com/google/javascript/jscomp/ClosureCodingConvention.java

@@ -194,7 +194,8 @@

         String qualifiedName = callee.getQualifiedName();
         if (functionName.equals(qualifiedName)) {
           Node target = callee.getNext();
-          if (target != null) {
+          if (target != null
+              && target.getType() == Token.STRING) {
             className = target.getString();
           }
         }
```

# Note

