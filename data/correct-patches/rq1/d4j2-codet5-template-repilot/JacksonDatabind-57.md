# Repilot Patch

```
return _bindAndReadValues(_considerFilter(_parserFactory.createParser(src, offset, length),
```

# Developer Patch

```
        return _bindAndReadValues(_considerFilter(_parserFactory.createParser(src, offset, length),
```

# Context

```
--- bug/JacksonDatabind-57/src/main/java/com/fasterxml/jackson/databind/ObjectReader.java

+++ fix/JacksonDatabind-57/src/main/java/com/fasterxml/jackson/databind/ObjectReader.java

@@ -1438,7 +1438,7 @@

         if (_dataFormatReaders != null) {
             return _detectBindAndReadValues(_dataFormatReaders.findFormat(src, offset, length), false);
         }
-        return _bindAndReadValues(_considerFilter(_parserFactory.createParser(src), 
+return _bindAndReadValues(_considerFilter(_parserFactory.createParser(src, offset, length),
                 true));
     }
```

# Note

