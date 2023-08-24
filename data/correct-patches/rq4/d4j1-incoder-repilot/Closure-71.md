# Repilot Patch

```
      boolean isOverride =
          parent.getJSDocInfo() != null &&
```

# Developer Patch

```
      boolean isOverride = parent.getJSDocInfo() != null &&
```

# Context

```
--- bug/Closure-71/src/com/google/javascript/jscomp/CheckAccessControls.java

+++ fix/Closure-71/src/com/google/javascript/jscomp/CheckAccessControls.java

@@ -413,7 +413,8 @@

     if (objectType != null) {
       // Is this a normal property access, or are we trying to override
       // an existing property?
-      boolean isOverride = t.inGlobalScope() &&
+      boolean isOverride =
+          parent.getJSDocInfo() != null &&
           parent.getType() == Token.ASSIGN &&
           parent.getFirstChild() == getprop;
```

# Note

