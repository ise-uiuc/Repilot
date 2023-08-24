# Repilot Patch

```
if (!result.isNoType()) {
```

# Developer Patch

```
    if (!result.isNoType()) {
```

# Context

```
--- bug/Closure-104/src/com/google/javascript/rhino/jstype/UnionType.java

+++ fix/Closure-104/src/com/google/javascript/rhino/jstype/UnionType.java

@@ -288,7 +288,7 @@

       builder.addAlternate(that);
     }
     JSType result = builder.build();
-    if (result != null) {
+if (!result.isNoType()) {
       return result;
     } else if (this.isObject() && that.isObject()) {
       return getNativeType(JSTypeNative.NO_OBJECT_TYPE);
```

# Note

