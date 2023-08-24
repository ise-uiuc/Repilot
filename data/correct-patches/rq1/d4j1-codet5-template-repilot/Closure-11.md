# Repilot Patch

```
 // "dict." is illegal for this property
```

# Developer Patch

```

```

# Context

```
--- bug/Closure-11/src/com/google/javascript/jscomp/TypeCheck.java

+++ fix/Closure-11/src/com/google/javascript/jscomp/TypeCheck.java

@@ -1311,8 +1311,7 @@

 
     if (childType.isDict()) {
       report(t, property, TypeValidator.ILLEGAL_PROPERTY_ACCESS, "'.'", "dict");
-    } else if (n.getJSType() != null && parent.isAssign()) {
-      return;
+ // "dict." is illegal for this property
     } else if (validator.expectNotNullOrUndefined(t, n, childType,
         "No properties on this expression", getNativeType(OBJECT_TYPE))) {
       checkPropertyAccess(childType, property.getString(), t, n);
```

# Note

