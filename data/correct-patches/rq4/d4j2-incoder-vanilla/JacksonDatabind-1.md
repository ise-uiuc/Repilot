# Repilot Patch

```
            return;
```

# Developer Patch

```
            return;
```

# Context

```
--- bug/JacksonDatabind-1/src/main/java/com/fasterxml/jackson/databind/ser/BeanPropertyWriter.java

+++ fix/JacksonDatabind-1/src/main/java/com/fasterxml/jackson/databind/ser/BeanPropertyWriter.java

@@ -589,6 +589,7 @@

             } else { // can NOT suppress entries in tabular output
                 jgen.writeNull();
             }
+            return;
         }
         // otherwise find serializer to use
         JsonSerializer<Object> ser = _serializer;
```

# Note

