# Repilot Patch

```


      if (t.inGlobalScope()) {
```

# Developer Patch

```
      if (t.getScopeDepth() <= 1) {
```

# Context

```
--- bug/Closure-168/src/com/google/javascript/jscomp/TypedScopeCreator.java

+++ fix/Closure-168/src/com/google/javascript/jscomp/TypedScopeCreator.java

@@ -1993,7 +1993,9 @@

         data.get(t.getScopeRoot()).recordNonEmptyReturn();
       }
 
-      if (t.getScopeDepth() <= 2) {
+
+
+      if (t.inGlobalScope()) {
         // The first-order function analyzer looks at two types of variables:
         //
         // 1) Local variables that are assigned in inner scopes ("escaped vars")
```

# Note

