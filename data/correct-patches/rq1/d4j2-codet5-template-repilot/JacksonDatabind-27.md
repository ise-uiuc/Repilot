# Repilot Patch

```
if (ext.handlePropertyValue(p, ctxt, propName, null)) {
```

# Developer Patch

```
                if (ext.handlePropertyValue(p, ctxt, propName, null)) {
```

# Context

```
--- bug/JacksonDatabind-27/src/main/java/com/fasterxml/jackson/databind/deser/BeanDeserializer.java

+++ fix/JacksonDatabind-27/src/main/java/com/fasterxml/jackson/databind/deser/BeanDeserializer.java

@@ -791,7 +791,7 @@

                 // first: let's check to see if this might be part of value with external type id:
                 // 11-Sep-2015, tatu: Important; do NOT pass buffer as last arg, but null,
                 //   since it is not the bean
-                if (ext.handlePropertyValue(p, ctxt, propName, buffer)) {
+if (ext.handlePropertyValue(p, ctxt, propName, null)) {
                     ;
                 } else {
                     // Last creator property to set?
```

# Note

