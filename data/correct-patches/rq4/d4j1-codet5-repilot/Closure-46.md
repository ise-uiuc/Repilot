# Repilot Patch

```

```

# Developer Patch

```

```

# Context

```
--- bug/Closure-46/src/com/google/javascript/rhino/jstype/RecordType.java

+++ fix/Closure-46/src/com/google/javascript/rhino/jstype/RecordType.java

@@ -137,22 +137,8 @@

         propertyNode);
   }
 
-  @Override
-  public JSType getLeastSupertype(JSType that) {
-    if (!that.isRecordType()) {
-      return super.getLeastSupertype(that);
-    }
-    RecordTypeBuilder builder = new RecordTypeBuilder(registry);
-    for (String property : properties.keySet()) {
-      if (that.toMaybeRecordType().hasProperty(property) &&
-          that.toMaybeRecordType().getPropertyType(property).isEquivalentTo(
-              getPropertyType(property))) {
-        builder.addProperty(property, getPropertyType(property),
-            getPropertyNode(property));
-      }
-    }
-    return builder.build();
-  }
+
+
   JSType getGreatestSubtypeHelper(JSType that) {
     if (that.isRecordType()) {
       RecordType thatRecord = that.toMaybeRecordType();
```

# Note

