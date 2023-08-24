# Repilot Patch

```
    @Override
    public JsonGenerator disable(Feature f) {
        super.disable(f);
        if (f == Feature.QUOTE_FIELD_NAMES) {
            _cfgUnqNames = true;
        }
        return this;
    }
```

# Developer Patch

```
    @Override
    public JsonGenerator disable(Feature f) {
        super.disable(f);
        if (f == Feature.QUOTE_FIELD_NAMES) {
            _cfgUnqNames = true;
        }
        return this;
    }
```

# Context

```
--- bug/JacksonCore-13/src/main/java/com/fasterxml/jackson/core/json/JsonGeneratorImpl.java

+++ fix/JacksonCore-13/src/main/java/com/fasterxml/jackson/core/json/JsonGeneratorImpl.java

@@ -124,6 +124,14 @@

         return this;
     }
 
+    @Override
+    public JsonGenerator disable(Feature f) {
+        super.disable(f);
+        if (f == Feature.QUOTE_FIELD_NAMES) {
+            _cfgUnqNames = true;
+        }
+        return this;
+    }
 
     @Override
     protected void _checkStdFeatureChanges(int newFeatureFlags, int changedFeatures) {
```

# Note

