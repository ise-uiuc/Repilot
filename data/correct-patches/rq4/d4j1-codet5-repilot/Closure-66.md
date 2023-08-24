# Repilot Patch

```
} else {
          typeable = false;
```

# Developer Patch

```
} else {
          typeable = false;
```

# Context

```
--- bug/Closure-66/src/com/google/javascript/jscomp/TypeCheck.java

+++ fix/Closure-66/src/com/google/javascript/jscomp/TypeCheck.java

@@ -513,7 +513,9 @@

         // Object literal keys are handled with OBJECTLIT
         if (!NodeUtil.isObjectLitKey(n, n.getParent())) {
           ensureTyped(t, n, STRING_TYPE);
-          // Object literal keys are not typeable
+
+        } else {
+          typeable = false;
         }
         break;
```

# Note

