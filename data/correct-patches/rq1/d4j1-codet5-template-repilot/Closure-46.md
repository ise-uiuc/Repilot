# Repilot Patch

```


  JSType buildPrototypeForRecord(JSType that) {
    RecordTypeBuilder builder = new RecordTypeBuilder(registry);
    for (String property : properties.keySet()) {
      if (that.toMaybeRecordType().hasProperty(property) &&
          that.toMaybeRecordType().getPropertyType(property).isEquivalentTo(
              getPropertyType(property))) {
        builder.addProperty(property, getPropertyType(property),
            getPropertyNode(property));
      }
    }
    return builder.build();
  }
```

# Developer Patch

```

```

# Context

```
--- bug/Closure-46/src/com/google/javascript/rhino/jstype/RecordType.java

+++ fix/Closure-46/src/com/google/javascript/rhino/jstype/RecordType.java

@@ -137,11 +137,9 @@

         propertyNode);
   }
 
-  @Override
-  public JSType getLeastSupertype(JSType that) {
-    if (!that.isRecordType()) {
-      return super.getLeastSupertype(that);
-    }
+
+
+  JSType buildPrototypeForRecord(JSType that) {
     RecordTypeBuilder builder = new RecordTypeBuilder(registry);
     for (String property : properties.keySet()) {
       if (that.toMaybeRecordType().hasProperty(property) &&
```

# Note

It's just a removal bug. So any newly defined method that is not used already is correct