# Repilot Patch

```
      if (fnType != null && fnType.hasInstanceType()) {
```

# Developer Patch

```
      if (fnType != null && fnType.hasInstanceType()) {
```

# Context

```
--- bug/Closure-125/src/com/google/javascript/jscomp/TypeCheck.java

+++ fix/Closure-125/src/com/google/javascript/jscomp/TypeCheck.java

@@ -1658,7 +1658,7 @@

     JSType type = getJSType(constructor).restrictByNotNullOrUndefined();
     if (type.isConstructor() || type.isEmptyType() || type.isUnknownType()) {
       FunctionType fnType = type.toMaybeFunctionType();
-      if (fnType != null) {
+      if (fnType != null && fnType.hasInstanceType()) {
         visitParameterList(t, n, fnType);
         ensureTyped(t, n, fnType.getInstanceType());
       } else {
```

# Note

